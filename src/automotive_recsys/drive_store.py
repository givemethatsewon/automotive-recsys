"""Private, file-scoped Google Drive storage for unattended experiment artifacts.

Credentials are supplied by a private file outside source/data bundles. Neither
credentials nor resumable-upload session URLs are written into experiment logs.
"""
import hashlib
import json
from pathlib import Path


def checksums(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            md5.update(block); sha.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "bytes": Path(path).stat().st_size}


class DriveStore:
    fields = "id,name,size,md5Checksum,webViewLink,modifiedTime,appProperties,parents"

    def __init__(self, credential_file, root_id=None, chunk_bytes=64 << 20):
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        self.credentials = Credentials.from_authorized_user_file(str(credential_file))
        self.service = build("drive", "v3", credentials=self.credentials, cache_discovery=False)
        self.root_id = root_id
        self.chunk_bytes = chunk_bytes
        self.folders = {"": root_id} if root_id else {}

    def create_root(self, name):
        result = self.service.files().create(body={"name": name, "mimeType": "application/vnd.google-apps.folder"}, fields="id,name,webViewLink").execute(num_retries=5)
        self.root_id = result["id"]; self.folders[""] = self.root_id
        return result

    def metadata(self, file_id):
        return self.service.files().get(fileId=file_id, fields=self.fields).execute(num_retries=5)

    def quota(self):
        return self.service.about().get(fields="storageQuota,user(emailAddress)").execute(num_retries=5)

    def children(self, folder_id, name=None):
        query = f"'{folder_id}' in parents and trashed = false"
        if name is not None:
            safe = name.replace("\\", "\\\\").replace("'", "\\'")
            query += f" and name = '{safe}'"
        result, token = [], None
        while True:
            response = self.service.files().list(q=query, fields=f"nextPageToken,files({self.fields},mimeType)", pageSize=1000, pageToken=token).execute(num_retries=5)
            result.extend(response.get("files", [])); token = response.get("nextPageToken")
            if not token: return result

    def folder(self, relative):
        relative = str(relative).strip("/")
        if relative in self.folders: return self.folders[relative]
        parent, _, name = relative.rpartition("/")
        parent_id = self.folder(parent)
        matches = self.children(parent_id, name)
        if len(matches) > 1: raise RuntimeError(f"Ambiguous experiment folder: {relative}")
        if matches:
            result = matches[0]
            if result["mimeType"] != "application/vnd.google-apps.folder": raise RuntimeError("A file occupies an experiment folder path.")
        else:
            result = self.service.files().create(body={"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}, fields="id").execute(num_retries=5)
        self.folders[relative] = result["id"]
        return result["id"]

    def find(self, relative):
        relative = str(relative).strip("/")
        parent, _, name = relative.rpartition("/")
        result = self.children(self.folder(parent), name)
        if len(result) > 1: raise RuntimeError(f"Ambiguous experiment file: {relative}")
        return result[0] if result else None

    def put(self, source, relative):
        from googleapiclient.http import MediaFileUpload
        source = Path(source); checksum = checksums(source)
        parent, _, name = str(relative).strip("/").rpartition("/")
        existing = self.find(relative)
        if existing and existing.get("md5Checksum") == checksum["md5"] and int(existing.get("size", -1)) == checksum["bytes"]:
            return existing | checksum
        media = MediaFileUpload(str(source), resumable=True, chunksize=self.chunk_bytes)
        body = {"name": name, "appProperties": {"sha256": checksum["sha256"]}}
        if existing:
            request = self.service.files().update(fileId=existing["id"], body=body, media_body=media, fields=self.fields)
        else:
            body["parents"] = [self.folder(parent)]
            request = self.service.files().create(body=body, media_body=media, fields=self.fields)
        response = None
        while response is None:
            _, response = request.next_chunk(num_retries=5)
        verified = self.metadata(response["id"])
        if verified.get("md5Checksum") != checksum["md5"] or int(verified.get("size", -1)) != checksum["bytes"]:
            raise IOError(f"Drive content verification failed: {relative}")
        return verified | checksum

    def put_json(self, value, relative, staging):
        staging = Path(staging); staging.mkdir(parents=True, exist_ok=True)
        path = staging / (hashlib.sha256(relative.encode()).hexdigest() + ".json")
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        return self.put(path, relative)

    def get_json(self, relative, default=None):
        item = self.find(relative)
        if not item: return default
        content = self.service.files().get_media(fileId=item["id"]).execute(num_retries=5)
        return json.loads(content)

    def download(self, file_id, destination, expected_sha256=None):
        from googleapiclient.http import MediaIoBaseDownload
        destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        with temporary.open("wb") as stream:
            request = self.service.files().get_media(fileId=file_id)
            download = MediaIoBaseDownload(stream, request, chunksize=self.chunk_bytes)
            finished = False
            while not finished: _, finished = download.next_chunk(num_retries=5)
        if expected_sha256 and checksums(temporary)["sha256"] != expected_sha256:
            raise IOError("Downloaded artifact SHA-256 differs from its durable manifest.")
        temporary.replace(destination)
        return destination

    def trash(self, relative):
        item = self.find(relative)
        if item:
            self.service.files().update(fileId=item["id"], body={"trashed": True}, fields="id").execute(num_retries=5)

    def remove_verified_checkpoint(self, file_id, expected_sha256):
        metadata = self.metadata(file_id)
        if not metadata["name"].endswith(".pt") or metadata.get("appProperties", {}).get("sha256") != expected_sha256:
            raise ValueError("Refusing to remove an unverified/non-checkpoint file.")
        self.service.files().delete(fileId=file_id).execute(num_retries=5)

    def remove_verified_artifact(self, file_id, expected_sha256):
        metadata = self.metadata(file_id)
        if metadata.get("appProperties", {}).get("sha256") != expected_sha256:
            raise ValueError("Refusing to remove an artifact without its verified content hash.")
        self.service.files().delete(fileId=file_id).execute(num_retries=5)

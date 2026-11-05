"""Bound unsuccessful recovery without counting productive runtime renewals."""
import re
import math


def effective_budget(plan, control):
    value = float(control.get("cu_budget", plan["cu_budget"]))
    if not math.isfinite(value) or value <= 0: raise ValueError("Invalid operational CU budget")
    return value


def owner_matches(config, owner):
    return not owner or config.get("controller_id") == owner.get("controller_id")


def progress_token(progress, previous=None):
    count = int(progress.get("completed_runs", 0))
    if previous and count < previous[0]: return previous
    match = re.match(r"^(.*?) epoch (\d+) \(", progress.get("durable_checkpoint") or "")
    if not match:
        return previous if previous and count == previous[0] else [count, "", 0]
    token = [count, match[1], int(match[2])]
    if previous and count == previous[0] and token[1] == previous[1] and token[2] <= previous[2]:
        return previous
    return token


def unsuccessful_workers(workers, token):
    count = 0
    for worker in reversed(workers):
        if worker.get("start_progress_token") != token: break
        count += 1
    return count


def transient_network_error(error):
    from requests.exceptions import RequestException
    from google.auth.exceptions import TransportError
    from httplib2 import HttpLib2Error
    if isinstance(error, (RequestException, TransportError, HttpLib2Error, TimeoutError, ConnectionError)):
        response = getattr(error, "response", None)
        if response is not None and response.status_code in (400, 401, 403, 404): return False
        return True
    status = getattr(getattr(error, "resp", None), "status", None)
    return status in (408, 429, 500, 502, 503, 504)

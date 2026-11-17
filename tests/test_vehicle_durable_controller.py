import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/colab_vehicle_durable.py'
spec=importlib.util.spec_from_file_location('durable_controller',SCRIPT)
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


class ControllerRecoveryTests(unittest.TestCase):
    def exercise(self,lost):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir();(root/'scripts').mkdir()
            cfg={'drive_root':'fixture','controller_id':'owner','runs':['fixture'],'context':'fixed','upload_parts':[],'session_prefix':'task','cu_budget':8}
            (run/'config.json').write_text(json.dumps(cfg))
            old={'name':'task-1','endpoint':'owned-old','ready':True,'launcher_pid':123,'start_progress_token':[0,0,0]}
            (run/'controller_state.json').write_text(json.dumps({'workers':[old],'token':[0,0,0],'initial_balance':100}))
            assigned={'foreign'} if lost else {'foreign','owned-old'};sessions={'task-1':SimpleNamespace(endpoint='owned-old')};released=[];launches=[]
            class Client:
                def list_assignments(self):return [SimpleNamespace(endpoint=x) for x in assigned]
                def get_consumption_user_info(self):return SimpleNamespace(paid_compute_units_balance=99,consumption_rate_hourly=1.94)
                def unassign(self,endpoint):released.append(endpoint);assigned.remove(endpoint)
            class States:
                def get(self,name):return sessions.get(name)
                def remove(self,name):sessions.pop(name,None)
            class Store:
                reads=0
                def get_json(self,path,default=None):
                    if path=='controller_owner.json':return {'controller_id':'owner'}
                    if path=='completion.json':
                        self.reads+=1;return {'status':'completed'} if self.reads>1 else None
                    if path=='progress.json':return {'durable_position':[0,3,1024]}
                    if path=='worker_heartbeat.json':
                        if lost:return {}
                        from datetime import datetime,timezone
                        return {'worker':'task-1','time':datetime.now(timezone.utc).isoformat(),'phase':'valid'}
                    if path.endswith('/completed.json'):return {'status':'completed','artifacts':[]}
                    return default
                def put_json(self,*a):pass
                def find(self,*a):return None
            def cli(*args,**kwargs):
                if args[0]=='new':
                    assigned.add('owned-new');sessions[args[2]]=SimpleNamespace(endpoint='owned-new')
                if args[0]=='download' and args[3].endswith('/prepared.json'):
                    Path(args[4]).write_text(json.dumps({'ready':True,'context':'fixed','kernel_unchanged':True}))
                return 0
            def popen(*a,**k):launches.append(a);return SimpleNamespace(pid=456)
            with patch.object(controller,'ROOT',root),patch.object(controller,'DriveStore',return_value=Store()),patch.object(controller,'cli',side_effect=cli),patch.object(controller.subprocess,'Popen',side_effect=popen),patch.object(controller.time,'sleep'),patch('colab_cli.client.Client',return_value=Client()),patch('colab_cli.state.StateStore',return_value=States()),patch('google.oauth2.credentials.Credentials.from_authorized_user_file',return_value=object()),patch('google.auth.transport.requests.AuthorizedSession',return_value=object()):
                controller.controller(run)
            state=json.loads((run/'controller_state.json').read_text())
            self.assertEqual(json.loads((run/'controller_status.json').read_text())['status'],'completed')
            self.assertNotIn('foreign',released);self.assertIn('foreign',assigned)
            self.assertEqual(len(launches),1 if lost else 0)
            self.assertEqual(state['token'],[0,3,1024] if lost else [0,3,1024])
            self.assertEqual(released,['owned-new'] if lost else ['owned-old'])

    def test_vm_loss_reallocates_and_preserves_foreign_session(self):self.exercise(True)
    def test_controller_restart_adopts_live_worker_without_duplicate_launch(self):self.exercise(False)


if __name__=='__main__':unittest.main()

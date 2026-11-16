import unittest
import numpy as np
from automotive_recsys.vehicle_v2 import compact_v2, match_v2


class VehicleV2Tests(unittest.TestCase):
    def test_missing_models_years_segments_universal_and_multiple_vehicles(self):
        def canon(make, model): return make, model
        def model_ok(a, b): return a is None or b is None or a == b
        users = [(0,'Honda','Civic','car',2010), (1,'Honda',None,'car',None),
                 (2,'Honda','CRF','powersports',2010), (0,'Ford','F150','car',2012)]
        items = [(0,'Honda','Civic','car',2009,2011,False),
                 (1,'Honda','Civic','car',2015,None,True),
                 (2,'Honda','CRF','powersports',2000,2020,False),
                 (3,'Ford','F150','car',2012,2012,False)]
        args = compact_v2(users,items,4,5,canon,model_ok)
        up,uv,ip,pf,compat=args
        ids=np.tile(np.arange(5),(4,1)); universal=np.array([False,True,False,False,False])
        state=match_v2(np.arange(4),ids,up,uv,ip,pf,universal,compat,True)
        np.testing.assert_array_equal(state,[[1,-1,0,1,-1],[1,1,0,0,-1],[0,-1,1,0,-1],[-1]*5])
        no_year=match_v2(np.arange(4),ids,up,uv,ip,pf,universal,compat,False)
        self.assertEqual(no_year[0,1],1)


if __name__ == '__main__': unittest.main()

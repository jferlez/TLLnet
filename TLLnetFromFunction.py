from TLLnet import TLLnet
import encapsulateLP
import region_helpers
import numpy as np
import numba as nb


class TLLnetFromFunction(TLLnet):

    def __init__(self, fn, eta=0.01, polytope=None):
        self.lp = encapsulateLP.encapsulateLP()
        pt = region_helpers.findInteriorPoint( np.hstack([-polytope[1],polytope[0]]), lpObj=self.lp )
        assert pt is not None, 'ERROR: specified polytope does not have an interior!'
        self.constraints = region_helpers.flipConstraintsReducedMin( polytope[0],polytope[1], fA=polytope[0], fb=polytope[1] )

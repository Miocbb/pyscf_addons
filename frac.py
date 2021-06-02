import numpy
from pyscf.scf import uhf

def _check(mf):
    if not isinstance(mf, uhf.UHF):
        raise RuntimeError('Cannot support this class of instance %s' % mf)


def frac(mf, occ):
    """
    SCF with fractional number of electrons for unrestricted HF or KS-DFT
    calculations.

    Parameters
    ----------
    occ : numpy.array
        The occupation number coresponding to the orbitals (in the increasing
        order w.r.t. orbital energies) for alpha and beta spins.
    """
    _check(mf)

    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        e_idx_a = numpy.argsort(mo_energy[0])
        e_idx_b = numpy.argsort(mo_energy[1])
        mo_occ = numpy.zeros_like(mo_energy)
        for i in range(len(occ[0])):
            mo_occ[0, e_idx_a[i]] = occ[0][i]
        for i in range(len(occ[1])):
            mo_occ[1, e_idx_b[i]] = occ[1][i]
        return mo_occ
    mf.get_occ = get_occ
    return mf


def frac_homo(mf, occ, spin):
    """
    SCF calculations with fractional occupation on HOMO with the specified spin.

    Parameters
    ----------
    occ : float
        The fractional occupation on HOMO.
    spin : [0, 1]
        The spin of HOMO. 0 for alpha and 1 for beta.

    See Also
    --------
    fraction()
    """
    _check(mf)

    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        e_idx_a = numpy.argsort(mo_energy[0])
        e_idx_b = numpy.argsort(mo_energy[1])
        n_a, n_b = mf.nelec
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[0, e_idx_a[:n_a]] = 1
        mo_occ[1, e_idx_b[:n_b]] = 1
        if spin == 0:
            if n_a > 0:
                mo_occ[0, e_idx_a[n_a-1]] = occ
            else:
                raise RuntimeError(
                    'No alpha electrons. Fail to set fractional occupation on alpha electron.')
        elif spin == 1:
            if n_b > 0:
                mo_occ[1, e_idx_a[n_a-1]] = occ
            else:
                raise RuntimeError(
                    'No beta electrons. Fail to set fractional occupation on beta electron.')
        return mo_occ
    mf.get_occ = get_occ
    return mf


def frac_lumo(mf, occ, spin):
    """
    SCF calculations with fractional occupation on LUMO with the specified spin.

    Parameters
    ----------
    occ : float
        The fractional occupation on HOMO.
    spin : [0, 1]
        The spin of LUMO. 0 for alpha and 1 for beta.
    """
    _check(mf)

    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        e_idx_a = numpy.argsort(mo_energy[0])
        e_idx_b = numpy.argsort(mo_energy[1])
        n_a, n_b = mf.nelec
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[0, e_idx_a[:n_a]] = 1
        mo_occ[1, e_idx_b[:n_b]] = 1

        if spin == 0:
            mo_occ[0, e_idx_a[n_a]] = occ
        elif spin == 1:
            mo_occ[1, e_idx_a[n_a]] = occ
        return mo_occ
    mf.get_occ = get_occ
    return mf

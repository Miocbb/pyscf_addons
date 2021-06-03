from pyscf_addons import frac
from pyscf.lib import logger
import numpy as np


def gsc_uks(mol, xc, frontier='homo', spin=0, step=1e-3):
    '''
    Perform UKS calculation using GSC with numerical 2nd order correction
    to calculate the GSC corrected frontier orbital energy.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule system.
    xc : str
        The pyscf supported xc functional.
    frontier : ['homo', 'lumo']
        The frontier orbital considered for the calculation.
    spin : [0, 1]
        The spin of the frontier orbital.
    step : float, default=1e-3
        The numerical step size that is used to numerical evaluation of the
        GSC 2nd order correction.

    Return
    ------
    mf_N : pyscf.dft.UKS()
        The pyscf SCF object for the integer N-electron system. Some new
        attributes are dynamically added into `mf_N`.

        Attributes
        ----------
        kappa : float
            The numerical 2nd order derivative of DFA energy w.r.t.
            the frontier orbital occupation number.
        gsc_orb_ene : float
            The GSC corrected frontier orbital energy in a.u.
        dfa_orb_ene : float
            The DFA frontier orbital energy in a.u.
        homo : [float, float]
            The HOMO energy in a.u. of alpha and beta spin.
        lumo : [float, float]
            The LUMO energy in a.u. of alpha and beta spin.
        gap : float
            The HOMO-LUMO gap in a.u.

    If the SCF fails to converged in the numerical evaluation,
    the return is None.
    '''
    E = []
    mf_N = None
    for i in range(3):
        if frontier == 'homo':
            frac_func = frac.frac_homo
            occ = 1 - step * i
        elif frontier == 'lumo':
            frac_func = frac.frac_lumo
            occ = 0 + step * i
        else:
            raise RuntimeError(f'Not a frontier orbital: {frontier}')

        mf = mol.UKS()
        mf.xc = xc
        mf = frac_func(mf, occ, spin)
        if mf.verbose >= logger.INFO:
            logger.info(mf, f'\n==> SCF running times: {i}')
        E.append(mf.scf())

        if not mf.converged:
            if mf.verbose >= logger.QUIET:
                logger.info(
                    mf, f'SCF not converged at step={i}. Cannot get numerical gsc correction.')
            return None

        if i == 0:
            mf_N = mf

    # get homo and lumo of the DFA
    e_idx_a = np.argsort(mf_N.mo_energy[0])
    e_idx_b = np.argsort(mf_N.mo_energy[1])
    e_sort_a = mf_N.mo_energy[0][e_idx_a]
    e_sort_b = mf_N.mo_energy[1][e_idx_b]
    na, nb = mf_N.nelec
    homo = (e_sort_a[na - 1], e_sort_b[nb - 1])
    lumo = (e_sort_a[na], e_sort_b[nb])

    # get GSC numerical curvature and GSC corrected orbital energy
    mf_N.kappa = (E[0] - 2 * E[1] + E[2]) / (step ** 2)
    mf_N.gsc_orb_ene = None
    mf_N.dfa_orb_ene = None
    if frontier == 'homo':
        mf_N.gsc_orb_ene = homo[spin] - 1.0 / 2 * mf_N.kappa
        mf_N.dfa_orb_ene = homo[spin]
    else:
        mf_N.gsc_orb_ene = lumo[spin] + 1.0 / 2 * mf_N.kappa
        mf_N.dfa_orb_ene = lumo[spin]
    mf_N.homo = homo
    mf_N.lumo = lumo
    mf_N.gap = min(lumo) - max(homo)

    if mf_N.verbose >= logger.INFO:
        chanel = 'alpha' if spin == 0 else 'beta'
        au2ev = 27.2116
        logger.info(mf_N, '\n==> GSC with numerical 2nd order correction <==')
        logger.info(mf_N, 'DFA gap: {:.8f} a.u. {:.8f} eV'.format(
            mf_N.gap, mf_N.gap * au2ev))
        logger.info(mf_N, 'DFA ({:s}-{:s}): {:.8f} a.u. {:.8f} eV'.format(
            chanel, frontier, mf_N.dfa_orb_ene, mf_N.dfa_orb_ene * au2ev))
        logger.info(mf_N, 'GSC ({:s}-{:s}): {:.8f} a.u. {:.8f} eV'.format(
            chanel, frontier, mf_N.gsc_orb_ene, mf_N.gsc_orb_ene * au2ev))
        logger.info(mf_N, '2nd order direvative ({:s}-{:s}): {:.8f}'.format(
            chanel, frontier, mf_N.kappa))

    return mf_N

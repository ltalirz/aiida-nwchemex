"""Utilities for writing NWChemEx input files."""
import pathlib
from attrs import define
import basis_set_exchange as bse
import qcelemental.models.molecule
import numpy as np

from .generic import BASIS_SET_OVERRIDES
from .nwchem import calc_basis
from ..protocols import recursive_merge

EXECUTABLES = ["HartreeFock", "CholeskyDecomp", "CD_CCSD", "CCSD_T"]


@define
class ProblemDefinition:  # pylint: disable=too-few-public-methods
    """Size of molecular chemistry problem."""

    nbasis: int = 0
    nelec: int = 0
    nocc: int = 0
    nvirt: int = 0

    @classmethod
    def from_molecule(
        cls, molecule: qcelemental.models.molecule, basis: str
    ) -> "ProblemDefinition":
        """Create a ProblemDefinition from a molecule and basis set."""
        s = cls()
        for symbol, Z in zip(molecule.symbols, molecule.atomic_numbers):
            element_basis = BASIS_SET_OVERRIDES.get(symbol, basis)
            bf = bse.get_basis(element_basis, elements=[Z], fmt="nwchem")
            s.nbasis = s.nbasis + calc_basis(bf)
            s.nelec = s.nelec + Z
        s.nelec = s.nelec + molecule.molecular_charge
        s.nocc = int(s.nelec / 2)
        s.nvirt = s.nbasis - s.nocc

        return s


def nwchemex_geometry(molecule: qcelemental.models.molecule, basis: str) -> dict:
    """Generate part of nwchemex json input file defined by geometry and basis set."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements

    multiplicity = molecule.molecular_multiplicity
    charge = molecule.molecular_charge

    geometry_key = {"coordinates": [], "units": "au"}
    geometry_key["coordinates"] = molecule.to_string(dtype="nwchem").splitlines()[1:-2]

    # basis specification
    basis_key = {"basisset": basis, "gaussian_type": "spherical"}
    special_elements = set(molecule.symbols).intersection(
        set(BASIS_SET_OVERRIDES.keys())
    )
    if special_elements:
        atom_basis = {}
        for element in special_elements:
            element_basis = BASIS_SET_OVERRIDES[element]
            atom_basis[element] = element_basis
            print(
                f"Warning: overriding basis '{basis}' to '{element_basis}' for element '{element}'."
            )

        basis_key["atom_basis"] = atom_basis

    return {
        "geometry": geometry_key,
        "basis": basis_key,
        "SCF": {
            "charge": charge,
            "multiplicity": multiplicity,
        },
    }


def nwchemex_ccsdt_input(
    molecule: qcelemental.models.molecule,
    basis: str,
    scf_type: str,
    settings: dict,
    executable: str,
):
    """Generate nwchemex json input file for given structure, basis set, and workflow definition."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    assert executable in EXECUTABLES, f"Unknown executable {executable}"

    dictionary = nwchemex_geometry(molecule=molecule, basis=basis)
    problem = ProblemDefinition.from_molecule(molecule=molecule, basis=basis)

    queue = settings["queue"]

    if scf_type in ("unrestricted", "uhf"):
        scf_type = "unrestricted"
    else:
        scf_type = "restricted"
    shMem = queue.ram_gb

    if executable in ["HartreeFock", "CholeskyDecomp"]:
        machines = 1
    elif executable == "CD_CCSD":
        machines = (
            (
                3 * problem.nvirt**4
                + 13 * (problem.nvirt**2) * (problem.nocc**2)
                + 5 * problem.nocc**4
            )
            * 8
            / 1e9
            / shMem
        )

        if scf_type == "restricted":
            machines /= 2
        else:
            machines *= 4

        machines = int(np.ceil(machines))
    elif executable == "CCSD_T":
        machines = int(28 * 8 * problem.nbasis**4 / 1e9 / shMem) + 1

    if executable == "HartreeFock":
        noscf = False
        restart = False
        readt = False
        writev = False
        writet = True
    else:
        noscf = True
        restart = True
        readt = True
        writev = False
        writet = True

    dictionary = recursive_merge(
        dictionary,
        {
            "SCF": {
                "noscf": noscf,
                "restart": restart,
                "scf_type": scf_type,
            },
            "CC": {
                "readt": readt,
                "writev": writev,
                "writet": writet,
                "CCSD(T)": {"ngpu": queue.n_gpus},
            },
        },
    )

    return machines, dictionary


# def nwchemex_ccsdt_inputs(
#     molecule: qcelemental.models.molecule,
#     basis: str,
#     scf_type: str,
#     step_settings: list,
# ):
#     """Generate nwchemex json input file for given structure, basis set, and workflow definition."""
#     # pylint: disable=too-many-locals,too-many-branches,too-many-statements

#     dictionary = nwchemex_geometry(molecule=molecule, basis=basis)
#     problem = ProblemDefinition(molecule=molecule, basis=basis)

#     inputDicts = []
#     stepMachines = []

#     if scf_type in ("unrestricted", "uhf"):
#         scf_type = "unrestricted"
#     else:
#         scf_type = "restricted"

#     for i_step, settings in enumerate(step_settings):
#         shMem = queue.ram_gb

#         if i_step in (1, 2):
#             machines = 1
#         elif i_step == 3:
#             machines = (
#                 (3 * nvirt**4 + 13 * (nvirt**2) * (nocc**2) + 5 * nocc**4)
#                 * 8
#                 / 1e9
#                 / shMem
#             )

#             if scf_type == "restricted":
#                 machines /= 2
#             else:
#                 machines *= 4

#             machines = int(np.ceil(machines))
#         else:
#             machines = int(28 * 8 * nbasis**4 / 1e9 / shMem) + 1

#         if i_step == 1:
#             noscf = False
#             restart = False
#             readt = False
#             writev = False
#             writet = True
#         else:
#             noscf = True
#             restart = True
#             readt = True
#             writev = False
#             writet = True

#         dictionary.update(
#             {
#                 "common": {"maxiter": 50},
#                 "SCF": {
#                     "noscf": noscf,
#                     "restart": restart,
#                     "scf_type": scf_type,
#                     "tol_int": 1e-20,
#                     "tol_lindep": 1e-5,
#                     "conve": 1e-08,
#                     "convd": 1e-07,
#                     "diis_hist": 10,
#                 },
#                 "CD": {"diagtol": 1e-06},
#                 "CC": {
#                     "threshold": 1e-06,
#                     "ccsd_maxiter": 100,
#                     "readt": readt,
#                     "writev": writev,
#                     "writet": writet,
#                     "CCSD(T)": {"ngpu": queue.n_gpus, "ccsdt_tilesize": 40},
#                 },
#             }
#         )
#         stepMachines.append(machines)
#         inputDicts.append(dictionary)

#     return stepMachines, inputDicts


def prepare_builder(settings, n_nodes, executable, code_label):
    """Prepare process builder for a step for any of our workflows.

    Populates calculation metadata, setting appropriate values for number of nodes, prepend text, etc.
    """
    from aiida_azq.registry import load_code  # pylint: disable=import-outside-toplevel

    queue = settings["queue"]

    code = load_code(
        label=code_label.format(executable=executable),
        template_vars=dict(executable=executable),
    )
    builder = code.get_builder()

    builder.metadata.options.resources = {
        "num_machines": n_nodes,
        # "num_cores_per_mpiproc": step_json["threads"],
        "num_mpiprocs_per_machine": queue.n_cores,
    }
    builder.metadata.options.max_wallclock_seconds = int(
        settings.get("wall_h", 24.0) * 3600
    )

    builder.metadata.options.queue_name = queue.queue_name
    # want `#SBATCH --get-user-env` to load correct modules
    builder.metadata.options.import_sys_environment = True

    if queue.n_gpus > 0:
        # For N GPUs, nwchemex wants N+1 MPI tasks
        builder.metadata.options.mpirun_extra_params = [
            "--map-by",
            f"ppr:{queue.n_gpus+1}:node",
        ]
        # set exclusive flag explicitly since we are using only some of the CPU
        builder.metadata.options.custom_scheduler_commands = "#SBATCH --exclusive"

    # if len(step_json["exclude"]) > 0:
    #     excludeStr = f"#SBATCH --exclude {step_json['exclude']}"
    #     if step_json["exclusive"]:
    #         builder.metadata.options.custom_scheduler_commands += "\n" + excludeStr
    #     else:
    #         builder.metadata.options.custom_scheduler_commands = excludeStr

    # TAMM uses /dev/shmem for shared memory (i.e. for all steps from HF to CCSD(T))
    # By default, /dev/shmem is 50% of RAM, below we resize it close to the full amount
    if pathlib.Path("/anfhome/.profile").exists():
        # empirically, a buffer of 10% or max. 20GB was found to suffice
        buffer_gb = min(queue.ram_gb * 0.1, 20)
        shmem_gb = queue.ram_gb - buffer_gb
        prepend_text = ""

        if n_nodes == 1:
            prepend_text += f"shmResize {shmem_gb}"
        else:
            prepend_text += f'\nsrun --nodes={n_nodes} --ntasks={n_nodes} --ntasks-per-node=1 bash -ic ". /anfhome/.profile; shmResize {shmem_gb}"'  # pylint: disable=line-too-long

        builder.metadata.options.prepend_text = prepend_text

    return builder

"""
Workflow for nwchem and nwchemex
"""
# pylint: disable=missing-function-docstring,missing-class-docstring
import pathlib

from aiida import orm, plugins
from aiida.engine import WorkChain, append_, while_
from aiida.engine.processes.exit_code import ExitCode
from aiida_nwchemex_cc.protocols import ProtocolMixin


class NwchemexCCChain(WorkChain, ProtocolMixin):
    """Run coupled cluster workflow with NWCHEMEX."""

    _CALCJOB_CLASS = plugins.CalculationFactory("nwchemex.nwchemexCalc")

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace(
            "steps",
            dynamic=True,
            help="Provice CalcJob input for each step, with ports 'step1', 'step2', etc.",
        )
        # for istep, step in enumerate(cls._STEPS):
        #     spec.expose_inputs(NwchemexCalculation, namespace=f"steps.step{istep+1}",
        #        exclude=('restart_folder',), namespace_options=dict(required=False))
        spec.input("group_label", valid_type=orm.Str, help="Group to save to, or empty")
        spec.outline(
            cls.setup,
            while_(cls.should_continue)(
                cls.run_step,
            ),
            cls.process_results,
        )

        spec.output_namespace(
            "results", dynamic=True, help="Collected results of all steps."
        )
        # spec.output("nodeIDs")

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return the ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        # pylint: disable=import-outside-toplevel
        from importlib_resources import files
        from . import protocols

        return files(protocols) / "base.yaml"

    @classmethod
    def get_builder_from_protocol(
        cls,
        molecule,
        basis: str,
        scf_type: str,
        step_settings: dict,
        protocol=None,
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param molecule: the ``qcelemental.models.molecule`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param relax_type: the relax type to use: should be a value of the enum ``common.types.RelaxType``.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param kwargs: additional keyword arguments that will be passed to the ``get_builder_from_protocol`` of all the
            sub processes that are called by this workchain.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        from aiida_nwchemex_cc.utils.nwchemex import (
            nwchemex_ccsdt_input,
            prepare_builder,
            EXECUTABLES,
        )  # pylint: disable=import-outside-toplevel

        builders = {}

        if len(step_settings) not in [4]:
            raise NotImplementedError("Expected 4 steps")

        for i_step, executable in enumerate(EXECUTABLES):
            settings = step_settings[executable]
            n_nodes, nwchemex_inputs = nwchemex_ccsdt_input(
                molecule=molecule,
                basis=basis,
                scf_type=scf_type,
                settings=settings,
                executable=executable,
            )
            builder = prepare_builder(
                settings=settings,
                n_nodes=n_nodes,
                executable=executable,
                code_label=settings["code"],
            )
            builder.parameters = orm.Dict(
                dict=cls.get_protocol_inputs(protocol, nwchemex_inputs)
            )
            builder.metadata.options.parser_name = "nwchemex_cc.nwchemex"
            builders[f"step{i_step+1}"] = builder

        builder = cls.get_builder()
        builder.steps = builders
        # builder.label = molecule.get_hash()
        # builder.description = f"CCSD(T) workflow for molecule {molecule.get_hash()}"

        return builder

    def setup(self):
        self.ctx.current_step = 1

    def should_continue(self):
        """Whether the WorkChain should run another step."""
        current_step = self.ctx.current_step

        if current_step > len(self.inputs.steps):
            return False

        # check that previous step ran fine
        if current_step > 1:
            if "results" not in self.ctx:
                raise ValueError(f"At step {current_step} but no results found.")
            if len(self.ctx.results) < current_step - 1:
                raise ValueError(
                    f"At step {current_step} but no results from step {current_step - 1} found."
                )
            previous_step = self.ctx.results[current_step - 2]
            if not previous_step.is_finished_ok:
                raise ValueError(f"Step {current_step-1} finished with errors.")

        return True

    def run_step(self):
        current_step = self.ctx.current_step
        inputs = self.inputs.steps[f"step{current_step}"]

        if current_step > 1:
            previous_step = self.ctx.results[current_step - 2]
            inputs["restart_folder"] = previous_step.outputs.remote_folder

            # # restarting t1/t2 amplitudes only works when sticking with CPU/GPU version
            # parameters = inputs["parameters"].get_dict()
            # if "CC" in parameters and "readt" in parameters["CC"] and parameters["CC"]["readt"]:
            #     current_code = inputs["code"].label
            #     previous_code = previous_step.inputs.code.label
            #     if 'gpu' in current_code.lower() and 'cpu' in previous_code.lower() or \
            #        'cpu' in current_code.lower() and 'gpu' in previous_code.lower():
            #        raise ValueError(f"Error at step {current_step}:"
            #  +" Cannot switch between CPU and GPU versions of NWCHEMEX when reading T1/T2 amplitudes.")

        # label of work chain is applied to calculation nodes as well
        inputs["metadata"]["label"] = self.node.label
        inputs["metadata"][
            "description"
        ] = f"Step {current_step} of NWChemEx workchain for structures {self.node.label}"

        future = self.submit(self._CALCJOB_CLASS, **inputs)
        self.ctx.current_step += 1
        return self.to_context(results=append_(future))

    def process_results(self):
        """Collect outputs from individual steps."""

        # collect results from all steps
        result_dict = {}
        for future in self.ctx.results:
            result_dict.update(future.outputs.results)

        self.out("results", result_dict)

        # Add WorkChain to the group
        group, _created = orm.Group.objects.get_or_create(
            label=self.inputs.group_label.value
        )
        group.add_nodes([self.node])
        return ExitCode(0)


class NwchemCCChain(NwchemexCCChain):
    """Run coupled cluster workflow with Nwchem."""

    _CALCJOB_CLASS = plugins.CalculationFactory("nwchemex_cc.nwchem")

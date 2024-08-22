# coding: utf-8

"""
Tasks to plot different types of histograms.
"""

from collections import OrderedDict
from abc import abstractmethod
# import matplotlib.pyplot as plt



import law
import luigi
import order as od

from columnflow.tasks.framework.base import Requirements, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, WeightProducerMixin,
    CategoriesMixin, ShiftSourcesMixin, HistHookMixin,
)
from columnflow.tasks.framework.plotting import (
    PlotBase, PlotBase1D, PlotBase2D, ProcessPlotSettingMixin, VariablePlotSettingMixin,
)
from columnflow.tasks.framework.decorators import view_output_plots
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms, MergeShiftedHistograms
from columnflow.tasks.data_driven_methods import DataDrivenEstimation
from columnflow.util import DotDict, dev_sandbox, dict_add_strict
from columnflow.plotting.plot_functions_ratio import plot_ratio_FF


class PlotVariablesBase(
    HistHookMixin,
    VariablePlotSettingMixin,
    ProcessPlotSettingMixin,
    CategoriesMixin,
    MLModelsMixin,
    WeightProducerMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))
    """sandbox to use for this task. Defaults to *default_columnar_sandbox* from
    analysis config.
    """

    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
        
    )
    """Set upstream requirements, in this case :py:class:`~columnflow.tasks.histograms.MergeHistograms`
    """

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for cat_name in sorted(self.categories)
            for var_name in sorted(self.variables)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["merged_hists"] = self.requires_from_branch()

        return reqs

    @abstractmethod
    def get_plot_shifts(self):
        return

    @law.decorator.log
    @view_output_plots
    def run(self):
        import hist
        from cmsdb.processes.qcd import qcd
        from cmsdb.processes.data import data

        # get the shifts to extract and plot
        plot_shifts = law.util.make_list(self.get_plot_shifts())
        hists_iso = {} # iso 
        hists_antiiso = {} # anti iso
        # prepare config objects
        # variable_tuple = self.variable_tuples[self.branch_data.variable]
        variable_tuple = self.variables
        print("variable_tuple", variable_tuple)
        variable_insts = [
            self.config_inst.get_variable(var_name)
            for var_name in variable_tuple
        ]
        print("variable_insts", variable_insts)
        for var in variable_insts:
            print("variable", var)
            # from IPython import embed; embed()
           
            categories = self.categories
            nb_cat = len(categories)
            for cat in categories:
                category_inst =self.config_inst.get_category(cat)
                print(self.config_inst.get_category(cat))
                # category_inst = self.config_inst.get_category(self.branch_data.category)
                leaf_category_insts = category_inst.get_leaf_categories() or [category_inst]
                print(leaf_category_insts)
                process_insts = list(map(self.config_inst.get_process, self.processes))
                print(process_insts)
                sub_process_insts = {
                    proc: [sub for sub, _, _ in proc.walk_processes(include_self=True)]
                    for proc in process_insts
                }
                # histogram data per process
                hists = {}
                with self.publish_step(f"plotting {var} in {category_inst.name}"):
                    qcd_hist_inp = self.input()['qcd_hists']
                    #from IPython import embed; embed()
                    # qcd_hist = qcd_hist_inp["collection"][0]['qcd_hists'].targets[self.branch_data.variable].load(formatter="pickle")
                    qcd_hist = qcd_hist_inp["collection"][0]['qcd_hists'].targets[var.name].load(formatter="pickle")

                    for dataset, inp in self.input().items():
                        if dataset != 'qcd_hists':
                            dataset_inst = self.config_inst.get_dataset(dataset)
                            # h_in = inp["collection"][0]["hists"].targets[self.branch_data.variable].load(formatter="pickle")
                            h_in = inp["collection"][0]["hists"].targets[var.name].load(formatter="pickle")

                            # loop and extract one histogram per process
                            for process_inst in process_insts:
                                # skip when the dataset is already known to not contain any sub process
                                if not any(map(dataset_inst.has_process, sub_process_insts[process_inst])):
                                    continue

                                # work on a copy
                                h = h_in.copy()

                                # axis selections
                                h = h[{
                                    "process": [
                                        hist.loc(p.id)
                                        for p in sub_process_insts[process_inst]
                                        if p.id in h.axes["process"]
                                    ],
                                    "category": [
                                        hist.loc(c.id)
                                        for c in leaf_category_insts
                                        if c.id in h.axes["category"]
                                    ],
                                    "shift": [
                                        hist.loc(s.id)
                                        for s in plot_shifts
                                        if s.id in h.axes["shift"]
                                    ],
                                }]

                                # axis reductions
                                h = h[{"process": sum, "category": sum}]

                                # add the histogram
                                if process_inst in hists:
                                    hists[process_inst] += h
                                else:
                                    hists[process_inst] = h
                    # there should be hists to plot
                    if not hists:
                        raise Exception(
                            "no histograms found to plot; possible reasons:\n" +
                            "  - requested variable requires columns that were missing during histogramming\n" +
                            "  - selected --processes did not match any value on the process axis of the input histogram",
                        )
                    # sort hists by process order
                    hists = OrderedDict(
                        (process_inst.copy_shallow(), hists[process_inst])
                        for process_inst in sorted(hists, key=process_insts.index)
                    )
                    # from IPython import embed; embed()
                    hists[qcd] = qcd_hist

                    print(">>>>>> category_inst", category_inst.name, "and variable", var.name)

                    if nb_cat ==2 : 
                        if "FFDRIso" in category_inst.name and var.name == "tau_1_pt":
                            print("fill numerator for variable", var.name)
                            hists_iso = hists[qcd]
                            print("hists_iso", hists_iso)
                        elif "FFDRantiIso" in category_inst.name and var.name == "tauantiiso_1_pt":
                            print("fill denominator for variable", var.name)
                            hists_antiiso = hists[qcd]
                            print("hists_antiiso", hists_antiiso)
                        else:
                            print("Not accounted in ratio plot")


                    # call the plot function
                    fig, _ = self.call_plot_func(
                        self.plot_function,
                        hists=hists,
                        config_inst=self.config_inst,
                        # category_inst=category_inst.copy_shallow(),
                        category_inst=category_inst.copy_shallow(),
                        # variable_insts=[var_inst.copy_shallow() for var_inst in variable_insts],
                        variable_insts=[var],
                        **self.get_plot_parameters(),
                    )

                    # save the plot
                    for outp in self.output()["plots"]:
                        print("outp", outp)
                        outp.dump(fig, formatter="mpl")
                    
                    for outp in self.custom_output(category_inst.name, var.name)["plots"]:
                        print("outp", outp)
                        outp.dump(fig, formatter="mpl")
                    

        # from IPython import embed; embed()
        if nb_cat == 2:
            print("Save ratio plots")

            for var_inst in variable_insts:
                if var_inst.name == "tau_1_pt":
                    variable_insts_iso = var_inst.copy_shallow()
                if var_inst.name == "tauantiiso_1_pt":
                    variable_insts_antiiso = var_inst.copy_shallow()

            # call the plot function
            fig, _ = plot_ratio_FF(
                config_inst = self.config_inst,
                hists_iso=hists_iso,
                hists_antiiso=hists_antiiso,
                category_inst_iso=category_inst.copy_shallow(),
                variable_insts_iso=variable_insts_iso,
                variable_insts_antiiso=variable_insts_antiiso,
                **self.get_plot_parameters(),
            )

            for outp in self.ratio_output("FFDRIso_tautau", "FFDRantiIso_tautau", variable_insts_iso.name, variable_insts_antiiso.name)["plots"]:
                print("outp", outp)
                outp.dump(fig, formatter="mpl")


            # hists_iso_1d = hists_iso.project("tau_1_pt")
            # hists_antiiso_1d = hists_antiiso.project("tauantiiso_1_pt")

            # print("hists_iso_1d", hists_iso_1d)
            # print("hists_antiiso_1d", hists_antiiso_1d)
            
            # # fig = plt.figure(figsize=(10, 8))

            # fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0), sharex=True)
            # main_ax_artists, sublot_ax_arists = hists_iso_1d.plot_ratio(
            #     hists_antiiso_1d,
            #     rp_ylabel=None,
            #     rp_num_label="hists_iso",
            #     rp_denom_label="hists_antiiso",
            # )

            plt.savefig("ratio.png")




class PlotVariablesBaseSingleShift(
    PlotVariablesBase,
    ShiftTask,
):
    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        PlotVariablesBase.reqs,
        MergeHistograms=MergeHistograms,
        DataDrivenEstimation=DataDrivenEstimation,
    )

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name})
            for var_name in sorted(self.variables)
            for cat_name in sorted(self.categories)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # no need to require merged histograms since each branch already requires them as a workflow
        if self.workflow == "local":
            reqs.pop("merged_hists", None)

        return reqs

    def requires(self):
        
        reqs = {
            d: self.reqs.MergeHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for d in self.datasets}
        
        reqs["qcd_hists"] = self.reqs.DataDrivenEstimation.req(self, branch=-1, categories={'FFDRantiIso_tautau'})
        return reqs
   
               

    def plot_parts(self) -> law.util.InsertableDict:
        parts = super().plot_parts()

        parts["processes"] = f"proc_{self.processes_repr}"
        parts["category"] = f"cat_{self.branch_data.category}"
        parts["variable"] = f"var_{self.branch_data.variable}"

        hooks_repr = self.hist_hooks_repr
        if hooks_repr:
            parts["hook"] = f"hooks_{hooks_repr}"

        return parts

    def output(self):
        return {
            "plots": [self.target(name) for name in self.get_plot_names("plot")],
        }

    def store_parts(self):
        parts = super().store_parts()
        if "shift" in parts:
            parts.insert_before("datasets", "shift", parts.pop("shift"))
        return parts

    # plot different varaibles and categories
    def custom_output(self, cat, var):
        b = self.branch_data
        return {"plots": [
            self.target(name)
            for name in self.get_plot_names(f"plot__proc_{self.processes_repr}__cat_{cat}__var_{var}")
        ]}

    # plot different varaibles and categories
    def ratio_output(self, cat_iso, cat_antiiso, var_iso, var_antiiso):
        b = self.branch_data
        return {"plots": [
            self.target(name)
            for name in self.get_plot_names(f"plot_ratio__proc_{self.processes_repr}__cat_{cat_iso}_vs_cat{cat_antiiso}__var_{var_iso}_vs_var_{var_antiiso}")
        ]}

    def get_plot_shifts(self):
        return [self.global_shift_inst]


class PlotVariables1D(
    PlotVariablesBaseSingleShift,
    PlotBase1D,
):
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_1d.plot_variable_per_process",
        add_default_to_description=True,
    )


class PlotVariables2D(
    PlotVariablesBaseSingleShift,
    PlotBase2D,
):
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_2d.plot_2d",
        add_default_to_description=True,
    )


class PlotVariablesPerProcess2D(
    law.WrapperTask,
    PlotVariables2D,
):
    # force this one to be a local workflow
    workflow = "local"

    def requires(self):
        return {
            process: PlotVariables2D.req(self, processes=(process,))
            for process in self.processes
        }


class PlotVariablesBaseMultiShifts(
    PlotVariablesBase,
    ShiftSourcesMixin,
):
    legend_title = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="sets the title of the legend; when empty and only one process is present in "
        "the plot, the process_inst label is used; empty default",
    )
    """
    """

    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        PlotVariablesBase.reqs,
        MergeShiftedHistograms=MergeShiftedHistograms,
    )

    def create_branch_map(self):
        return [
            DotDict({"category": cat_name, "variable": var_name, "shift_source": source})
            for var_name in sorted(self.variables)
            for cat_name in sorted(self.categories)
            for source in sorted(self.shift_sources)
        ]

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # no need to require merged histograms since each branch already requires them as a workflow
        if self.workflow == "local":
            reqs.pop("merged_hists", None)

        return reqs

    def requires(self):
        return {
            d: self.reqs.MergeShiftedHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for d in self.datasets
        }

    def plot_parts(self) -> law.util.InsertableDict:
        parts = super().plot_parts()

        parts["processes"] = f"proc_{self.processes_repr}"
        parts["shift_source"] = f"unc_{self.branch_data.shift_source}"
        parts["category"] = f"cat_{self.branch_data.category}"
        parts["variable"] = f"var_{self.branch_data.variable}"

        hooks_repr = self.hist_hooks_repr
        if hooks_repr:
            parts["hook"] = f"hooks_{hooks_repr}"

        return parts

    def output(self):
        return {
            "plots": [self.target(name) for name in self.get_plot_names("plot")],
        }

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("datasets", "shifts", f"shifts_{self.shift_sources_repr}")
        return parts

    def get_plot_shifts(self):
        return [
            self.config_inst.get_shift(s) for s in [
                "nominal",
                f"{self.branch_data.shift_source}_up",
                f"{self.branch_data.shift_source}_down",
            ]
        ]

    def get_plot_parameters(self):
        # convert parameters to usable values during plotting
        params = super().get_plot_parameters()
        dict_add_strict(params, "legend_title", None if self.legend_title == law.NO_STR else self.legend_title)
        return params


class PlotShiftedVariables1D(
    PlotBase1D,
    PlotVariablesBaseMultiShifts,
):
    plot_function = PlotBase.plot_function.copy(
        default="columnflow.plotting.plot_functions_1d.plot_shifted_variable",
        add_default_to_description=True,
    )


class PlotShiftedVariablesPerProcess1D(law.WrapperTask):

    # upstream requirements
    reqs = Requirements(
        PlotShiftedVariables1D.reqs,
        PlotShiftedVariables1D=PlotShiftedVariables1D,
    )

    def requires(self):
        return {
            process: self.reqs.PlotShiftedVariables1D.req(self, processes=(process,))
            for process in self.processes
        }
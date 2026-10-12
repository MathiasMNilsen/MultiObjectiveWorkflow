"""The full reservoir model: renders the Mako deck, runs OPM Flow and reads the summary vectors."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import subprocess

from mako.template import Template
from resdata.summary import Summary
from tqdm import tqdm

PRODUCER_VECTORS = ("WOPR", "WGPR", "WWPR")  # oil, gas and water production rate
INJECTOR_VECTORS = ("WWIR", "WTHP")          # water injection rate and tubing-head pressure


class Simulator:
    """Simulates daily well controls over one control interval, starting from the current restart state.

    Every run gets its own folder in `<output_dir>/simulations`. A run started with `keep=<name>` also writes
    restart files. `accept(<name>)` then makes its final state the initial state of the next interval. The
    first interval starts from a restart file of the history run.
    """

    def __init__(self, deck, model_dir, history_case, restart_step, output_dir, producers, injectors,
                 flow="flow", n_parallel=1):
        self.template = Template(filename=str(deck))
        # Runs are named after the history case (not the template, whose name may differ in case), since OPM
        # Flow writes its output files in upper case and the restart files are copied from that case
        self.case = Path(history_case).name.upper()
        self.wells = [*producers, *injectors]
        self.keys = ([f"{v}:{w}" for w in producers for v in PRODUCER_VECTORS]
                     + [f"{v}:{w}" for w in injectors for v in INJECTOR_VECTORS])
        self.flow, self.n_parallel = flow.split(), n_parallel
        self.work_dir = Path(output_dir) / "simulations"
        self.restart_dir = Path(output_dir) / "restarts"
        self.restart_dir.mkdir(parents=True)
        for step in (0, restart_step):  # OPM Flow also reads step 0 of the case it restarts from
            shutil.copy(f"{history_case}.X{step:04d}", self.restart_dir / f"{self.case}.X{step:04d}")
        self.restart_step, self.first_interval = restart_step, True

        # Paths in the deck are relative to the run folders (all at the same depth) to keep lines short
        run_dir = self.work_dir / "run"
        self.paths = {"model_dir": os.path.relpath(model_dir, run_dir),
                      "restart_root": os.path.relpath(self.restart_dir / self.case, run_dir)}

    def run(self, controls, dates, keep=None):
        """Simulate control schedules in parallel.

        Each schedule is an (n_wells, n_days) array of daily rates [Sm3/day] for the report dates `dates`.
        Returns one dict per schedule, mapping summary keys such as 'WOPR:A1' to daily values, or None if
        the run failed. With `keep=<name>`, the single run is stored under that name, restart files included.
        """
        names = [keep] if keep else [f"run_{n:03d}" for n in range(len(controls))]
        pool = ThreadPoolExecutor(self.n_parallel)
        try:
            jobs = [pool.submit(self._run, u, name, dates, keep is not None) for u, name in zip(controls, names)]
            return [job.result() for job in tqdm(jobs, desc="  simulations", ncols=80, leave=False,
                                                 disable=len(jobs) == 1)]
        finally:
            pool.shutdown(cancel_futures=True)  # on Ctrl+C, do not start the queued runs

    def accept(self, name):
        """Continue the next interval from the final state of the kept run `name`."""
        last = max((self.work_dir / name).glob(f"{self.case}.X*"), key=lambda f: int(f.suffix[2:]))
        last.replace(self.restart_dir / last.name)
        self.restart_step, self.first_interval = int(last.suffix[2:]), False
        shutil.rmtree(self.work_dir)

    def _run(self, u, name, dates, save_restart):
        folder = self.work_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        deck = self.template.render(rates=dict(zip(self.wells, u.tolist())), dates=dates,
                                    restart_step=self.restart_step, first_interval=self.first_interval,
                                    save_restart=save_restart, **self.paths)
        (folder / f"{self.case}.DATA").write_text(deck)

        response = None
        if subprocess.run([*self.flow, f"{self.case}.DATA"], cwd=folder, capture_output=True).returncode == 0:
            try:
                frame = Summary(str(folder / self.case), include_restart=False).pandas_frame(
                    time_index=list(dates.to_pydatetime()), column_keys=self.keys)
                response = {key: frame[key].to_numpy() for key in self.keys}
            except OSError:  # the summary files could not be read
                pass
        if not save_restart:
            shutil.rmtree(folder)
        return response

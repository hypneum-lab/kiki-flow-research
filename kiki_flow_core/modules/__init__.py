"""Coupled PDE modules: advection-diffusion, scaffolding scheduler, phonological loop."""

from kiki_flow_core.modules.advection_diffusion import AdvectionDiffusion
from kiki_flow_core.modules.phonological_loop import PhonologicalLoop
from kiki_flow_core.modules.scaffolding_scheduler import ScaffoldingScheduler

__all__ = ["AdvectionDiffusion", "PhonologicalLoop", "ScaffoldingScheduler"]

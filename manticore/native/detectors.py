import hashlib
import logging
from typing import Optional
from contextlib import contextmanager

import capstone
from prettytable import PrettyTable

from ..core.plugin import Plugin
from ..core.smtlib import Constant, Operators, issymbolic

logger = logging.getLogger(__name__)


class DetectorClassification:
    """
    Description of detector event.
    """

    HIGH = 0
    MEDIUM = 1
    LOW = 2
    INFORMATIONAL = 3


classification_txt = {
    DetectorClassification.INFORMATIONAL: "Informational",
    DetectorClassification.LOW: "Low",
    DetectorClassification.MEDIUM: "Medium",
    DetectorClassification.HIGH: "High",
}


def get_detectors_classes():
    return [DetectArbitraryControlFlowRedirect]


def output_detectors(detector_classes):
    """
    Prints a visually appealing list of supported native binary detectors.
    """
    detectors_list = []

    for detector in detector_classes:
        argument = detector.ARGUMENT
        help_info = detector.HELP
        impact = detector.IMPACT
        confidence = classification_txt[detector.CONFIDENCE]
        detectors_list.append((argument, help_info, impact, confidence))

    table = PrettyTable(["Num", "Check", "What it Detects", "Impact", "Confidence"])

    # Sort by impact, confidence, and name
    detectors_list = sorted(
        detectors_list, key=lambda element: (element[2], element[3], element[0])
    )
    idx = 1
    for (argument, help_info, impact, confidence) in detectors_list:
        table.add_row([idx, argument, help_info, classification_txt[impact], confidence])
        idx = idx + 1

    print(table)


class Detector(Plugin):
    ARGUMENT: Optional[
        str
    ] = None  # argument that needs to be passed to --detect to use given detector
    HELP: Optional[str] = None  # help string
    IMPACT: Optional[int] = None  # DetectorClassification value
    CONFIDENCE: Optional[int] = None  # DetectorClassification value

    @property
    def name(self):
        return self.__class__.__name__.split(".")[-1]

    def get_findings(self, state):
        return state.context.setdefault(f"{self.name}.findings", set())

    @contextmanager
    def locked_global_findings(self):
        with self.manticore.locked_context(f"{self.name}.global_findings", set) as global_findings:
            yield global_findings

    @property
    def global_findings(self):
        with self.locked_global_findings() as global_findings:
            return global_findings

    def add_finding(self, state, pc, finding, constraint=True):
        """
        Logs a finding at specified contract and assembler line.
        :param state: Current state
        :param pc: Program counter of the finding
        :param finding: Textual description of the finding
        :param constraint: Finding is considered reproducible only when constraint is True
        """

        if isinstance(pc, Constant):
            pc = pc.value
        if not isinstance(pc, int):
            raise ValueError("PC must be a number")
        self.get_findings(state).add((pc, finding, constraint))
        with self.locked_global_findings() as gf:
            gf.add((pc, finding))
        logger.warning(finding)

    def add_finding_here(self, state, finding, constraint=True):
        """
        Logs a finding in current contract and assembler line.
        :param state: Current state
        :param finding: Textual description of the finding
        :param constraint: Finding is considered reproducible only when constraint is True
        """
        pc = state.platform.PC
        if isinstance(pc, Constant):
            pc = pc.value
        if not isinstance(pc, int):
            raise ValueError("PC must be a number")
        self.add_finding(state, pc, finding, constraint)

    def _save_current_location(self, state, finding, condition=True):
        """
        Save current location in the internal locations list and returns a textual id for it.
        This is used to save locations that could later be promoted to a finding if other conditions hold
        See _get_location()
        :param state: Current state
        :param finding: Textual description of the finding
        :param condition: General purpose constraint
        """
        pc = state.platform.PC
        location = (pc, finding, condition)
        hash_id = hashlib.sha1(str(location).encode()).hexdigest()
        state.context.setdefault(f"{self.name}.locations", {})[hash_id] = location
        return hash_id

    def _get_location(self, state, hash_id):
        """
        Get previously saved location.

        A location is composed of: pc, finding, condition
        """
        return state.context.setdefault("{:s}.locations".format(self.name), {})[hash_id]


def pc_unconstrained_heuristics(state, instruction: capstone.CsInsn) -> bool:
    """
    Heuristics for the symbolic PC register to determine whether the user has "significant" control over redirection.

    The definition of "significant" is not scientific and is as follows:
        - Arbitrary address that likely won't be a normal valid target (0x5)

    :param state: The current state of the execution
    :param instruction: The instruction that made PC symbolic
    :return: True if sufficiently unconstrained
    """
    # Check if we can reach an arbitrary target
    # 5 is chosen because it is small enough to fit in all architecture PC register sizes
    arbitrary_target = 0x5
    if state.can_be_true(state.cpu.PC == arbitrary_target):
        return True

    # Check if the instruction that is calling the symbolic PC can be used for dynamic control-flow redirection.
    # Examples of dynamic redirection instructions are `ret`, `call r/m64`, etc. --- Instructions that do not have
    # explicit destinations.
    # TODO(ekilmer): Figure out how to get a list of these specific instructions in an architecture dependent way.
    if instruction.mnemonic.lower() in ["call", "ret"]:
        return True

    # Else our heuristics don't match
    return False


class DetectArbitraryControlFlowRedirect(Detector):
    """
    Detect when PC could execute at an arbitrary symbolic address.

    This could mean that a buffer has overwritten an address that is being used for execution and may be
    possible to exploit.
    """

    ARGUMENT = "arb-cf-redirect"
    HELP = "A potential, arbitrary, user-controlled control-flow redirection"
    IMPACT = DetectorClassification.HIGH
    CONFIDENCE = DetectorClassification.MEDIUM

    def __init__(self, target_address_list=None):
        """
        Detect a symbolic PC register that might be controllable with user input to an arbitrary location.

        :param target_address_list: Initialize this detector with an optional target address list to check for
        redirection.
        """
        super().__init__()
        self.target_address_list = target_address_list if target_address_list else []

    def did_execute_instruction_callback(self, state, pc, target_pc, instruction):
        """
        Determine whether this instruction can take advantage of a potentially vulnerable control-flow redirection.

        TODO(ekilmer) This might require taint analysis to determine whether user-controlled input actually appears in
            the destination address in an arbitrary manner (jump tables are untested and may cause trouble).
        :param state: Current state of Manticore
        :param pc: Old PC
        :param target_pc: New PC
        :param instruction: Old instruction
        :return: pass
        """
        if issymbolic(target_pc) and pc_unconstrained_heuristics(state, instruction):
            finding_message = (
                f"Previous PC was concrete (0x{pc:x}); new PC is symbolic. "
                f"Instruction was {instruction.mnemonic.upper()}."
            )

            # Generate a testcase to be inspected later
            self.manticore.generate_testcase(state, finding_message, name="detector")

            # Check to see if we can reach targets
            pc_expr = state.cpu.PC
            reachable_targets = []
            for target in self.target_address_list:
                # Exploit if possible
                if state.can_be_true(pc_expr == target):
                    finding_message += f" Can reach target 0x{target:x}"
                    reachable_targets.append(target)
                else:
                    finding_message += f" Cannot reach target 0x{target:x}"

            self.add_finding(state, pc, finding_message)

            # Fork to target addresses
            if len(reachable_targets) > 0:
                # Add new constraints to guide execution flow to target addresses
                if len(reachable_targets) >= 2:
                    expr = Operators.OR(*map(lambda x: pc_expr == x, reachable_targets))
                else:  # len(reachable_targets) == 1:
                    expr = pc_expr == reachable_targets[0]
                state.constrain(expr)

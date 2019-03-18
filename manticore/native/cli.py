from .detectors import DetectArbitraryControlFlowRedirect
from .manticore import Manticore
from ..core.plugin import InstructionCounter, Visited, Tracer, RecordSymbolicBranches


def choose_detectors(args):
    from .detectors import get_detectors_classes
    all_detector_classes = get_detectors_classes()
    detectors = {d.ARGUMENT: d for d in all_detector_classes}
    arguments = list(detectors.keys())

    detectors_to_run = []

    if not args.exclude_all:
        exclude = []

        if args.detectors_to_exclude:
            exclude = args.detectors_to_exclude.split(',')

            for e in exclude:
                if e not in arguments:
                    raise Exception(f'{e} is not a detector name, must be one of {arguments}. See also `--list-detectors`.')

        for arg, detector_cls in detectors.items():
            if arg not in exclude:
                detectors_to_run.append(detector_cls)

    return detectors_to_run


def native_main(args, _logger):
    env = {key: val for key, val in [env[0].split('=') for env in args.env]}

    m = Manticore(args.argv[0], argv=args.argv[1:], env=env, entry_symbol=args.entrysymbol,
                  workspace_url=args.workspace, policy=args.policy,
                  concrete_start=args.data, pure_symbolic=args.pure_symbolic)

    # Default plugins for now.. FIXME REMOVE!
    m.register_plugin(InstructionCounter())
    m.register_plugin(Visited())
    m.register_plugin(Tracer())
    m.register_plugin(RecordSymbolicBranches())

    # Enable detectors
    for detector in choose_detectors(args):
        m.register_detector(detector())

    # Fixme(felipe) remove this, move to plugin
    m.coverage_file = args.coverage

    if args.names is not None:
        m.apply_model_hooks(args.names)

    if args.assertions:
        m.load_assertions(args.assertions)

    @m.init
    def init(initial_state):
        for file in args.files:
            initial_state.platform.add_symbolic_file(file)

    m.run(procs=args.procs, should_profile=args.profile)

    for detector in list(m.detectors):
        m.unregister_detector(detector)

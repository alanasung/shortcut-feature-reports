def test_pkg():
    import introspect.introspect as d
    assert d.FEATURES

def test_modules():
    from introspect.introspect import activations, causal, data, probes, train, verbalize, pipeline
    assert callable(pipeline.stage_build_dataset)

def test_stages_mod():
    from introspect import stages
    assert callable(stages.STAGES["report"])

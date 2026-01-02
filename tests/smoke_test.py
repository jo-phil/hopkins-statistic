from hopkins_statistic import hopkins

statistic = hopkins([[x] for x in range(3)], rng=42)
if 0 < statistic < 1:
    print("Smoke test succeeded")  # noqa: T201
else:
    msg = f"Smoke test failed; hopkins returned {statistic}"
    raise RuntimeError(msg)

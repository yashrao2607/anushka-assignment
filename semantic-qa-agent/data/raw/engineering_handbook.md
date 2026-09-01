# Engineering Handbook

## Code Review Standards

Every change to a production repository requires review and approval by at least
one other engineer. Reviews should be completed within one working day. A change
that alters authentication, payment, or data-retention behaviour requires a
second reviewer drawn from the security team.

## Branching and Release

Work happens on short-lived feature branches cut from main. Releases are cut
weekly on Wednesday. Hotfixes may be released at any time with approval from the
on-call lead and must be merged back to main the same day.

## Testing Expectations

New code paths require automated tests. Pull requests that reduce overall test
coverage are blocked by the pipeline. Flaky tests must be fixed or quarantined
within one week of being reported. Silently disabling a test is not acceptable.

## On-Call Rotation

The on-call rotation runs weekly, handing over on Monday morning. The on-call
engineer acknowledges a critical alert within 15 minutes and provides a written
incident summary within 24 hours of resolution.

## Production Access

Direct write access to production databases is not granted to individuals.
Changes are applied through reviewed migrations. Emergency read access may be
granted for a maximum of four hours and is logged and reviewed afterwards.

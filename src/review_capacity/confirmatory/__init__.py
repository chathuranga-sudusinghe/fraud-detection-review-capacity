"""The confirmatory Test run: record contracts and the binding of the frozen artifacts (Issue #195).

Nothing in this package reads the Test period. The contracts validate the records of the single
authorised Test evaluation; the binding checks verify, before any Test row is read, that the
artifacts it will use are the frozen ones. The Test command itself is a separate issue.
"""

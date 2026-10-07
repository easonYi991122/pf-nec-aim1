"""Child-local GAIN selection for the new, reduced-menu team interfaces.

Pass an already masked FeatureRows object. The caller owns grouped splits,
the fit ledger, resource guard and approval; no data are loaded here.
"""
from pf_nec import contract as c, features as f, selectors as sel


def gain50(train_rows, train_labels, context, **ranking_options):
    """Rank only the current training child and return its frozen selection."""
    if train_rows.bank != c.D5_BANK or tuple(train_rows.values) != c.D5SAFE:
        raise c.ContractError("GAIN requires the complete masked D5-safe training bank")
    ranking = sel.gain_rank(train_rows, train_labels, context, **ranking_options)
    columns = [entry["column"] for entry in ranking["ranking"][:50]]
    c.feature_columns("GAIN50", frame=train_rows.frame, selected=columns)
    selection = dict(frame=train_rows.frame, bank=c.D5_BANK, recipe="GAIN50",
                     columns=columns, fit_binding=ranking["binding"], context=ranking["context"])
    return selected_rows(train_rows, selection), selection


def selected_rows(rows, selection):
    """Apply the train-fitted column list without reading labels or reranking."""
    if (selection["frame"] != rows.frame or selection["bank"] != c.D5_BANK
            or rows.bank != c.D5_BANK or selection["recipe"] != "GAIN50"):
        raise c.ContractError("Selection frame/bank/recipe differs")
    columns = c.feature_columns("GAIN50", frame=rows.frame, selected=selection["columns"])
    c.assert_predictors(columns, bank=rows.bank, dependencies=rows.dependencies)
    return f.select_features(rows, columns)

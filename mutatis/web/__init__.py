"""The read-only web inspector: an HTTP server over a results store (`server`), the answers it gives as plain data (`describe`
for the specification, `graph` for the graph of blocks and its layout, `estimates` for priors against posteriors, fit and
quantities, `compare` for two runs side by side), and the page that draws them (`static/`).

The top layer: it may use every other layer, and nothing imports it except the command line (`python -m mutatis inspect`).
"""

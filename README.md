# blg-py

This is the back-end for BPMN Learn & Grade (BLG)

# Installing dependencies

First install the dependencies necessary by running:

``pip install .``

Followed by

``python -m spacy download en_core_web_md``

(uv equivalent: ``uv run -- spacy download en_core_web_md``)

To download the necessary pipeline for natural language processing.

This project makes use of parts of the ``rust_bpmn_analzyer`` for control flow analysis and in order to use that
dependency it has to be built for which the rust toolchain must be installed.

Once installed, the command ``maturin develop`` must be called in order to compile the Rust code into a functional
Python library.

# Running this application

Once all the dependencies are installed the tool can be ran as follows:

``py main.py example``

This will use the example folder and its subdirectory ``submissions`` for storing the submission BPMN models.

Now, an API will be exposed on localhost that can be interfaced with
using [blg-web](https://github.com/ETolboom/blg-web).


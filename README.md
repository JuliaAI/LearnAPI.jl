# MLInterface.jl

An Julia interface for training and applying models in machine learning and statistics


&#x1F6A7;

| Linux | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/MLInterface.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLInterface.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/MLInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLInterface.jl?branch=master) |

**Status.** Proposal stage (no code)

This repository is to provide a general purpose machine learning interface. It is designed
based on experiences of developers of MLJ's [MLJModelInterface.jl]() which it will
eventually replace, but hopes to be useful more generally. The design is in a state of flux
and comments (posted as issues) are welcome.

The interface makes wide use of traits to articulate model functionality. There is no
abstract model type heirarchy. Model data type requirements can be articulated using
[scientific types](https://github.com/JuliaAI/ScientificTypes.jl) but this is optional.




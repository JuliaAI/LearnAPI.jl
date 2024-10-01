ERR_NOT_NAME_VALUE_EXPRESSION(head) = ArgumentError(
    "Expected `var=value` expression in @trait macro call but got `$head`-expression "*
    "instead. "
)

function name_value_pair(ex)
    ex.head == :(=) || throw(ERR_NOT_NAME_VALUE_EXPRESSION(ex.head))
    return (ex.args[1], ex.args[2])
end

"""
    @trait(TypeEx, trait1=value1, trait2=value2, ...)

Overload a number of traits for algorithms of type `TypeEx`. For example, the code

```julia
@trait(
    RidgeRegressor,
    tags = ("regression", ),
    doc_url = "https://some.cool.documentation",
)
```

is equivalent to

```julia
LearnAPI.tags(::RidgeRegressor) = ("regression", ),
LearnAPI.doc_url(::RidgeRegressor) = "https://some.cool.documentation",
```

"""
macro trait(algorithm_ex, exs...)
    program = quote end
    for ex in exs
        trait_ex, value_ex = name_value_pair(ex)
        push!(
            program.args,
            :($LearnAPI.$trait_ex(::$algorithm_ex) = $value_ex),
        )
    end
    return esc(program)
end

function is_uppercase(char::Char)
    i = Int(char)
    i > 64 && i < 91
end

# """
#     snakecase(str, del='_')

# Return the snake case version of the abstract string or symbol, `str`, as in

#     snakecase("TheLASERBeam") == "the_laser_beam"

# """
function snakecase(str::AbstractString; delim='_')
    snake = Char[]
    n = length(str)
    for i in eachindex(str)
        char = str[i]
        if is_uppercase(char)
            if i != 1 && i < n &&
                !(is_uppercase(str[i + 1]) && is_uppercase(str[i - 1]))
                push!(snake, delim)
            end
            push!(snake, lowercase(char))
        else
            push!(snake, char)
        end
    end
    return join(snake)
end

snakecase(s::Symbol) = Symbol(snakecase(string(s)))

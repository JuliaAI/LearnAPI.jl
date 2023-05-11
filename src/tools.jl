ERR_NOT_NAME_VALUE_EXPRESSION(head) = ArgumentError(
    "Expected `var=value` expression in @trait macro call but got `$head`-expression "*
    "instead. "
)

function name_value_pair(ex)
    ex.head == :(=) || throw(ERR_NOT_NAME_VALUE_EXPRESSION(ex.head))
    return (ex.args[1], ex.args[2])
end

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

"""

    typename(T::Type)

Return a symbol corresponding to the name of the type `T`, stripped of
any type-parameters and module qualifications. For example:

    _typename(MLJBase.Machine{MLJAlgorithms.ConstantRegressor,true})

returns `:Machine`. Where this does not make sense (eg, instances of
`Union`) `Symbol(string(M))` is returned.

"""
function typename(M)
    if isdefined(M, :name)
        return M.name.name
    elseif isdefined(M, :body)
        return typename(M.body)
    else
        return Symbol(string(M))
    end
end

function is_uppercase(char::Char)
    i = Int(char)
    i > 64 && i < 91
end

"""
    snakecase(str, del='_')

Return the snake case version of the abstract string or symbol, `str`, as in

    snakecase("TheLASERBeam") == "the_laser_beam"

"""
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

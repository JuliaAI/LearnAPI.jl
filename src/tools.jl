ERR_NOT_NAME_VALUE_EXPRESSION(head) = ArgumentError(
    "Expected `var=value` expression in @trait macro call but got `$head`-expression "*
    "instead. "
)

function name_value_pair(ex)
    ex.head == :(=) || throw(ERR_NOT_NAME_VALUE_EXPRESSION(ex.head))
    return (ex.args[1], ex.args[2])
end

macro trait(model_ex, exs...)
    program = quote end
    for ex in exs
        trait_ex, value_ex = name_value_pair(ex)
        push!(
            program.args,
            :($LearnAPI.$trait_ex(::Type{<:$model_ex}) = $value_ex),
        )
    end
    return esc(program)
end

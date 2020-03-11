module Sars

export SARS, fill_q!

using Test: @test

mutable struct SARS{S, A}
    s  :: S
    a  :: A
    r  :: Float32
    q  :: Union{Nothing, Float32}
    s′ :: S
    f  :: Bool
end

function fill_q!(sars; discount_factor=1)
    q = 0
    for i in length(sars):-1:1
        q *= discount_factor
        if sars[i].f
            q = 0
        end
        q += sars[i].r
        sars[i].q = q
    end
end

function test_fill_q()
    sars = [
        SARS(nothing, nothing, 1f0, nothing, nothing, false),
        SARS(nothing, nothing, 1f0, nothing, nothing, false),
        SARS(nothing, nothing, 1f0, nothing, nothing, true),
        SARS(nothing, nothing, 1f0, nothing, nothing, false),
        SARS(nothing, nothing, 1f0, nothing, nothing, true)]
    fill_q!(sars)
    @test sars[1].q == 3
    @test sars[2].q == 2
    @test sars[3].q == 1
    @test sars[4].q == 2
    @test sars[5].q == 1
end

function test_all()
    test_fill_q()
end

test_all()

end # module

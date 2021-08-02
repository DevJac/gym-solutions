# To run: julia --project make_J.jl
using PackageCompiler
create_sysimage([
    :DataStructures,
    :Flux,
    :LearnBase,
    :OpenAIGym,
    :PyCall,
    :Zygote,
], sysimage_path="J")

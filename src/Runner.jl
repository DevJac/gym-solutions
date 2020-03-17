module Runner

export environment, statetype, actiontype, train!
export run_episodes, batch_train_until_reward!

using BSON
using Distributed
using OpenAIGym
using Plots
using Printf
using ProgressMeter
using Sars

environment(policy) = error("unimplemented")
statetype(policy) = error("unimplemented")
actiontype(policy) = error("unimplemented")
train_policy!(policy, sars) = error("unimplemented")

function run_episodes(n_episodes, policy; render_count=0, kwargs...)
    sars = SARS{statetype(policy), actiontype(policy)}[]
    rewards = Float32[]
    presults = if render_count < 1 && nprocs() > 1
        @showprogress "Env batch: " pmap(1:n_episodes) do _
            run_episodes′(1, policy, render_count=render_count, parallel=true, kwargs...)
        end
    else
        progress = Progress(n_episodes, "Env batch: ")
        map(1:n_episodes) do _
            r = run_episodes′(1, policy, render_count=render_count, parallel=false, kwargs...)
            next!(progress)
            r
        end
    end
    for (psars, prewards) in presults
        append!(sars, psars)
        append!(rewards, prewards)
    end
    sars, rewards
end

function run_episodes′(n_episodes, policy; render_count=0, close_env=false, parallel=false)
    env = environment(policy)
    if parallel
        run_env = GymEnv(env.name, env.ver)
    else
        run_env = env
    end
    sars = SARS{statetype(policy), actiontype(policy)}[]
    rewards = Float32[]
    for episode in 1:n_episodes
        reward = run_episode(run_env, policy) do (s, a, r, s′)
            if parallel
                s = copy(s)
                a = copy(a)
                r = copy(r)
                s′ = copy(s′)
            end
            push!(sars, SARS{statetype(policy), actiontype(policy)}(s, a, r, nothing, s′, finished(run_env)))
            if render_count >= episode; render(run_env) end
        end
        push!(rewards, reward)
    end
    if close_env; close(env) end
    sars, rewards
end

last(xs, n) = xs[max(1, end-n+1):end]

clear_lines(n) = print("\u1b[F\u1b[2K" ^ n)

function batch_train_until_reward!(policy, stop_reward; batch_size=100, fancy_output=false, save_policy=false)
    try
        print("\n" ^ 4)
        start_time = time()
        all_rewards = Float32[]
        summary_rewards = []
        means = Tuple{Float32, Float32}[]
        for training_iteration in Iterators.countfrom()
            iteration_batch_size = max(batch_size, training_iteration)
            sars, rewards = run_episodes(iteration_batch_size, policy)
            if fancy_output; run_episodes(1, policy, render_count=1) end
            append!(all_rewards, rewards)
            recent_rewards = last(all_rewards, iteration_batch_size)
            push!(summary_rewards, summarystats(recent_rewards))
            push!(means, (length(all_rewards), summary_rewards[end].mean))
            if save_policy
                bson(@sprintf("policy/policy_%03d.bson", training_iteration),
                     all_rewards=all_rewards,
                     summary_rewards=summary_rewards,
                     means=means,
                     policy=policy)
            end
            clear_lines(4)
            @printf("%3d: Time: %4.2f    Best Mean: %8.3f    Mean: %8.3f    IQR: %8.3f, %8.3f, %8.3f\n",
                    training_iteration, (time() - start_time) / 60^2, maximum(s.mean for s in summary_rewards),
                    summary_rewards[end].mean,
                    summary_rewards[end].q25, summary_rewards[end].median, summary_rewards[end].q75)
            if fancy_output
                pyplot()
                scatter(all_rewards, size=(1200, 800), background_color=:black, markercolor=:white, legend=false,
                        markersize=3, markeralpha=0.3,
                        markerstrokewidth=0, markerstrokealpha=0)
                plot!(means, linecolor=:red,
                      linewidth=1, linealpha=0.5)
                display(scatter!(means,
                                 markercolor=:red, markershape=:vline,
                                 markersize=11, markeralpha=0.2,
                                 markerstrokewidth=0, markerstrokealpha=0))
            end
            if mean(recent_rewards) >= stop_reward; break end
            train_policy!(policy, sars)
        end
    catch e
        if typeof(e) != InterruptException; rethrow() end
    finally
        close(environment(policy))
    end
    policy
end

end # module

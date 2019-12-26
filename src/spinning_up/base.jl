using Distributed
using Flux
using OpenAIGym
using Plots
using Printf
using Statistics

pyplot()

Base.iterate(set::DiscreteSet) = iterate(set.items)
Base.iterate(set::DiscreteSet, state) = iterate(set.items, state)
Flux.onehot(x, set::DiscreteSet) = Flux.onehot(x, set.items)
OpenAIGym.sample(set::DiscreteSet, weights) = sample(set.items, weights)

struct SARS
    s
    a
    r
    s′
    f
end

function run_episodes_parallel(n_episodes, policy)
    sars = []
    rewards = []
    futures = map(1:n_episodes) do _
        @spawnat :any run_episodes(1, policy, parallel=true)
    end
    for future in futures
        future_sars, future_rewards = fetch(future)
        append!(sars, future_sars)
        append!(rewards, future_rewards)
    end
    sars, rewards
end

function run_episodes(n_episodes, policy; render_count=0, close_env=false, parallel=false)
    if parallel
        env = GymEnv(policy.env.name, policy.env.ver)
    else
        env = policy.env
    end
    sars = []
    rewards = []
    for episode in 1:n_episodes
        reward = run_episode(env, policy) do (s, a, r, s′)
            if parallel
                s = copy(s)
                a = copy(a)
                r = copy(r)
                s′ = copy(s′)
            end
            push!(sars, SARS(s, a, r, s′, finished(env)))
            if render_count >= episode; render(env) end
        end
        push!(rewards, reward)
    end
    sars, rewards
end

last(xs, n) = xs[max(1, end-n+1):end]

struct RunStats
    wall_time
    episodes
    training_iterations
end

function run_until_reward(policy, stop_reward)
    start_time = time()
    all_rewards = []
    mean_rewards = Tuple{Int64, Float64}[]  # Type annotation is needed for plots.
    batch_size = 100
    best_mean_reward = -Inf
    training_iteration = 0
    try
        while true
            training_iteration += 1
            @printf("%2.2f %4d: batch size: %3d  ", (time() - start_time) / (60 * 60), training_iteration, batch_size)
            sars, rewards = run_episodes_parallel(batch_size, policy)
            append!(all_rewards, rewards)
            recent_rewards = last(all_rewards, 100)
            push!(mean_rewards, (length(all_rewards), mean(recent_rewards)))
            if length(recent_rewards) == 100
                if mean(recent_rewards) > best_mean_reward
                    best_mean_reward = mean(recent_rewards)
                    batch_size = max(10, floor(batch_size * 0.8))
                else
                    batch_size += 1
                end
            end
            @printf("episodes: %5d  recent rewards: %7.2f  best reward: %7.2f\n", length(all_rewards), mean(recent_rewards), best_mean_reward)
            scatter(all_rewards, size=(1200, 800), markercolor=:blue, legend=false,
                    markersize=3, markeralpha=0.3,
                    markerstrokewidth=0, markerstrokealpha=0)
            plot!(mean_rewards, linecolor=:red,
                  linewidth=1, linealpha=0.5)
            display(scatter!(mean_rewards,
                             markercolor=:red, markershape=:vline,
                             markersize=11, markeralpha=0.2,
                             markerstrokewidth=0, markerstrokealpha=0))
            sleep(0.001)  # This enables the plot to update immediately.
            if mean(recent_rewards) >= stop_reward; break end
            policy.train_policy!(policy, sars)
        end
    catch e
        if typeof(e) != InterruptException; rethrow() end
    end
    policy, RunStats(time() - start_time, length(all_rewards), training_iteration)
end

function print_stats_summary(stats)
    @printf("Wall Time          : %6d ± %6d\n", mean(s.wall_time for s in stats), std(s.wall_time for s in stats))
    @printf("Episodes           : %6d ± %6d\n", mean(s.episodes for s in stats), std(s.episodes for s in stats))
    @printf("Training Iterations: %6d ± %6d\n", mean(s.training_iterations for s in stats), std(s.training_iterations for s in stats))
end

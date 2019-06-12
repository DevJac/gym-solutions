# Turns out, a basic linear transformation can make a great policy.
# Inspired by: https://github.com/udacity/deep-reinforcement-learning/blob/master/hill-climbing/Hill_Climbing.ipynb
module CartPole

using OpenAIGym
using Printf
import Reinforce: action

const env = GymEnv(:CartPole, :v1)
const state_size = 4
const action_size = 2

struct Policy <: AbstractPolicy
    weights
end

function action(policy::Policy, r, s, A)
    # Julia uses 1-based indexing, so argmax will return 1 or 2,
    # but the valid actions in this environment are 0 and 1,
    # so we subtract 1 from whatever argmax returns.
    argmax(policy.weights * s) - 1
end

# A function that does nothing,
# because we must pass a function to run_episode,
# and we don't care what that function does.
noop(_) = nothing

function hill_climb()
    # The weights are a linear transformation from state to action.
    best_weights = zeros(action_size, state_size)
    all_rewards = []
    for episode in Iterators.countfrom()
        # If we haven't succeeded by episode 120, let's try something completely new.
        episode_weights = episode % 120 == 0 ? randn(action_size, state_size) : randn(action_size, state_size) .+ best_weights
        episode_reward = run_episode(noop, env, Policy(episode_weights))
        push!(all_rewards, episode_reward)
        if episode_reward >= maximum(all_rewards)
            best_weights = episode_weights
        end
        recent_rewards = length(all_rewards) >= 100 ? all_rewards[end-99:end] : all_rewards
        @printf("Episode: %3d    Mean of recent rewards: %3.0f\n", episode, mean(recent_rewards))
        if episode >= 100 && mean(recent_rewards) > 195
            return best_weights
        end
    end
end

function run()
    solution_weights = hill_climb()
    println("Observe the solution")
    for _ in 1:10
        run_episode(env, Policy(solution_weights)) do _
            render(env)
        end
    end
end

end # end module

if !isinteractive()
    CartPole.run()
end

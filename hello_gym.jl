import PyCall

gym = PyCall.pyimport("gym")
env = gym.make("LunarLander-v2")

for episode in 1:5
    steps = 0
    total_rewards = 0
    env.reset()
    env.render()
    done = false
    while !done
        a = rand(0:3)
        (s′, r, done, _) = env.step(a)
        env.render()
        steps += 1
        total_rewards += r
    end
    @info "Episode $episode finished" episode steps total_rewards
end

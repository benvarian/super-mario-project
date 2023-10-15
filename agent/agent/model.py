def stuff():
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from gym.wrappers import frame_stack, GrayScaleObservation
    from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
    from matplotlib import pyplot as plt
    from stable_baselines3 import PPO

    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    env = gym_super_mario_bros.make(
        "SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human"
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")

    model = PPO.load("./train/rl_model_PPO25_6000000.zip")
    obs = env.reset()
    done = False
    total_reward = 0.0
    amount_reward = 0
    max_x_pos = 0
    start = 400
    end = 10000
    data = []
    action_list = []
    location_list = []
    while not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        amount_reward += 1
        x = info[0]["x_pos"]
        location_list.append((info[0]["x_pos"], info[0]["y_pos"]))
        # print(action[0])
        time = info[0]["time"]
        action_list.append(action[0])
        if x > max_x_pos:
            max_x_pos = x
        if time < end:
            end = time
        # if x > 1500:
        #     time_taken-=info[0]["time"]
        #     # data.append(time_taken)
        #     # print(f"max_x_pos: {max_x_pos}")
        # else:
        #     # data.append(0)
        env.render()
    env.close()
    # file.write(
    #     f"iter:{i}, reward:{total_reward}, amount of rewards:{amount_reward}, average reward:{total_reward / amount_reward}\n"
    # )
    # if x > 1500:
    #     return 1
    # else:
    #     return 0
    data.append(action_list)
    data.append(location_list)
    data.append(end)
    if max_x_pos >= 1500:
        return data
    else:
        return None

    # return f"completed: {done}, reward:{total_reward[0]},max_x_pos: {max_x_pos}, amount of rewards:{amount_reward}, average reward:{total_reward / amount_reward}\n"


def handleArray(arr):
    import csv

    actions = arr[0]
    x_y = arr[1]
    time_left = arr[2]
    # print(len(x_y))
    with open("actions.csv", "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["id", "action_1", "action_2"])
        for i, action in enumerate(actions):
            writer.writerow([i, action])
    with open("x_y.csv", "w", newline="") as output:
        writer = csv.writer(output, quotechar='"', quoting=csv.QUOTE_NONE)
        writer.writerow(["id", "x_1", "y_1", "x_2", "y_2"])
        data = []
        length = len(x_y[0])
        for i in range(length):
            line = str(f"{x_y[0][i]},{x_y[1][i]}")
            arr = list(line)
            if "(" in arr:
                arr.remove("(")
                arr.remove(")")
                arr.remove("(")
                arr.remove(")")
                arr.remove(" ")
                arr.remove(" ")
            string = str("".join(arr))

            writer.writerow([i, string])
            # line.replace("(", "")
            # line.replace(")", "")
            # print(line)
        # print(x_y[0][1000], x_y[1][1000])
        # for items in x_y:
        #     temp = []
        #     for i, (x, y) in enumerate(items):
        #         temp.append([x, y])
        #     # data.append(temp)
        #     print(temp)
        # print(data)

    # with open("time_left.csv", "w", newline="") as output:
    #     writer = csv.writer(output)
    #     writer.writerow(["id", "time_left"])
    #     writer.writerow([0, time_left])


with open("data.txt", "w") as file:
#     # file.write("2-1\n")
    actions = []
    x_y = []
    time_left = []
    z = 20
    while z > 0:
        print(z)
        res = stuff()
        if res is None:
            z = z
        else:
            actions.append(res[0])
            x_y.append(res[1])
            time_left.append(res[2])
            z -= 1
    data = [actions, x_y, time_left]
    file.write(str(data))
# file.write(str(i))
# print(actions)
# file.write(res)

# arr = [[4, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 4, 4, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 2, 2, 4, 4, 2, 4, 5, 2, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 2, 4, 4, 2, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 5, 5, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 2, 2, 5, 1, 1, 1, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 4, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 6, 2, 6, 5, 6, 5, 5, 5, 2, 6, 6, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 0, 6, 2, 6, 0, 0, 6, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 2, 0, 0, 4, 4, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 4, 2, 0, 4, 4, 4, 2, 4, 2, 4, 5, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 2, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 5, 2, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 4, 4, 4, 4, 4, 0, 4, 6, 0, 6, 0, 4, 4, 2, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 2, 2, 2, 2, 5, 2, 5, 5, 5, 5, 5, 5, 2, 5, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 2, 0, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 5, 2, 2, 5, 0, 5, 5, 0, 5, 0, 0, 5, 5, 5, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 4, 5, 5, 6, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 6, 5, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 6, 0, 2, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [4, 1, 1, 1, 4, 4, 1, 1, 4, 4, 4, 1, 4, 1, 4, 4, 4, 1, 1, 4, 4, 1, 4, 4, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 2, 2, 2, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 5, 2, 2, 2, 2, 2, 2, 2, 5, 2, 5, 4, 2, 4, 4, 4, 2, 4, 4, 4, 4, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 4, 4, 2, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 0, 5, 4, 2, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 5, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 0, 2, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 2, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 0, 4, 4, 4, 4, 0, 0, 6, 0, 2, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 2, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2, 5, 5, 5, 5, 5, 0, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 4, 4, 4, 6, 6, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 5, 2, 2, 2, 6, 2, 5, 2, 5, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 6, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]]
# def handletest(arr):
#     final = []
#     length = len(arr)
#     print(length)
#     max_len = 0
#     for item in arr:
#         length2 = len(item)
#         if length2 > max_len:
#             max_len = length2
#         print(length2)
#     for i in range(max_len):
#         arr0 = None
#         if i < len(arr[0]):
#             arr0 = arr[0][i]
#         else:
#             arr0 = None
#         arr1 = None
#         if i < len(arr[1]):
#             arr1 = arr[1][i]
#         else:
#             arr1 = None
#         # if i > len(arr[1])
#         print(i,arr0,arr1)
            
# handletest(arr)

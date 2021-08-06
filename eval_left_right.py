import gym
import argparse
import itertools
import torch as T
import sys
import os

from glob import glob

def BulitIn(net, path, STEP, STEP_MUL, END_IDX, BI_CNT, is_right):
    env = gym.make('gym_pikachu_volleyball:pikachu-volleyball-v0',
            isPlayer1Computer=is_right, isPlayer2Computer=(not is_right))

    isPlayer2Serve = False

    observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

    elos = [1200] * ((END_IDX - STEP * STEP_MUL - 1) // STEP // STEP_MUL + 1)

    for idx, V in enumerate(range(STEP * STEP_MUL, END_IDX, STEP * STEP_MUL)):    
        if is_right:
            net.load_state_dict(T.load(path + '{:09d}_R.pt'.format(V)))
        else:
            net.load_state_dict(T.load(path + '{:09d}_L.pt'.format(V)))

        gameCount, frameCount = 0, 0

        while True:
            with T.no_grad(): 
                p, *_ = net(observation[0])

            # a = T.argmax(p)
            a = T.distributions.Categorical(p).sample()

            if is_right:
                observation, reward, done, _ = env.step((0, a))
            else:
                observation, reward, done, _ = env.step((a, 0))

            observation = T.tensor(observation, dtype=T.float32)

            frameCount += 1

            if frameCount >= 9000: done = True; reward = 0.5;

            if done: 
                isPlayer2Serve = not isPlayer2Serve
                observation = T.tensor(env.reset(isPlayer2Serve), dtype=T.float32)

                frameCount = 0; gameCount += 1;

                if is_right:
                    reward = (reward + 1) / 2
                else:
                    reward = (1 - reward) / 2

                _, elos[idx] = updateELO(1200, elos[idx], 1 - reward, reward)

                if gameCount >= BI_CNT: break
        print(f'Version {V}, ELO: {elos[idx]}')
        sys.stdout.flush()

    env.close()
    print()
    sys.stdout.flush()
    return elos

def Eval(RS, END_IDX, STEP, STEP_MUL, BI_CNT):
    RS.sort()

    N, nets, paths = len(RS), [], []

    for R in RS:
        path = 'Saves/Run{:02d}/'.format(R)

        net = __import__('Saves.Run{:02d}.Model'.format(R), fromlist=[None]).Model()
        net.eval()

        path = path + 'Models/'

        nets.append(net)
        paths.append(path)
        
        print('Eval For LEFT Start!')
        sys.stdout.flush()
        elo = BulitIn(net, path, STEP, STEP_MUL, END_IDX, BI_CNT, False)
        filename1 = f'Results/ELO-{R}-LEFT.txt'
        os.makedirs(os.path.dirname(filename1))
        with open(filename1, 'w') as f:
            for e in elo: f.write(f'{e}\n')

        print('Eval For RIGHT Start!')
        sys.stdout.flush()
        elo = BulitIn(net, path, STEP, STEP_MUL, END_IDX, BI_CNT, True)
        filename2 = f'Results/ELO-{R}-RIGHT.txt'
        os.makedirs(os.path.dirname(filename2))
        with open(filename2, 'w') as f:
            for e in elo: f.write(f'{e}\n')

def updateELO(RA, RB, SA, SB):
    EA = 1 / (1 + 10 ** ((RB - RA) / 400))
    EB = 1 / (1 + 10 ** ((RA - RB) / 400))
    return RA + 32 * (SA - EA), RB + 32 * (SB - EB)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--runs', nargs='*', type=int)

    parser.add_argument('--end', type=int, default=int(5e7))
    parser.add_argument('--step', type=int, default=160000)
    parser.add_argument('--step_mul', type=int, default=1)
    parser.add_argument('--bi_cnt', type=int, default=100)

    args = parser.parse_args()
    Eval(args.runs, args.end, args.step, args.step_mul, args.bi_cnt)


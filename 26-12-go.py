#+ d'infos dans alphazero/nochi/cyril.txt

import alphazero.nochi.michi as nochi

if __name__ == '__main__':
    net = nochi.GoModel(load_snapshot=None)
    #net = nochi.GoModel(load_snapshot="alphazero/nochi/G171107T013304_000000150")
    nochi.selfplay(net)
    net.server.terminate()
    net.server.join()
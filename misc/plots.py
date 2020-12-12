"""
    Some plots
"""

import matplotlib.pyplot as plt

fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()


def plot_graph(x, y1, y2, y3, xl="games", l1="scores", l2="bombs", l3="idle"):
    global fig, host, par1, par2
    par2.spines["right"].set_position(("axes", 1.2))
    par2.set_frame_on(True)
    par2.patch.set_visible(False)
    for sp in par2.spines.values():
        sp.set_visible(False)

    par2.spines["right"].set_visible(True)

    p1, = host.plot(x, y1, "C1", label=l1)
    p2, = par1.plot(x, y2, "C2", label=l2)
    p3, = par2.plot(x, y3, "C3", label=l3)

    host.set_xlabel(xl)
    host.set_ylabel(l1)
    par1.set_ylabel(l2)
    par2.set_ylabel(l3)

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [lin.get_label() for lin in lines])
    plt.ion()
    plt.show()
    plt.pause(0.05)

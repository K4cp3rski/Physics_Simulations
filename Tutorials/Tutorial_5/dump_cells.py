import matplotlib.pyplot as plt

"""
Note that I'm assuming that this funcitons is inside of class
which contains 
*position vector named r = [[x,y],...,[]]
*radius for each particles named R = [...]
*force for each particle named force = [[x,y],...,[]]
"""

def dump(filename, r, R, force, L, plot_forces=False):
    fig, ax = plt.subplots()
    for i in range(0, len(r)):
        ax.add_patch(plt.Circle((r[i][0], r[i][1]), R[i], facecolor='green', edgecolor='black'))
        if plot_forces:
            ax.quiver(r[i][0], r[i][1], force[i][0], force[i][1], angles='xy', scale_units='xy',
                      scale=0.8, color='black')
    plt.axis('equal')
    plt.axis('on')
    plt.tight_layout()
    plt.axis([0,L,0,L])
    plt.savefig('motion/{}.png'.format(filename))
    plt.close()
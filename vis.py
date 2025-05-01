import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_two_sequences(
    seq1, seq2, title1="Sequence 1", title2="Sequence 2", interval=200
):
    """
    Display two sequences of frames side by side with titles.

    :param seq1: tensor of shape [Channels, Time, Height, Width]
    :param seq2: tensor of shape [Channels, Time, Height, Width]
    :param title1: title for the seq1.
    :param title2: title for the seq2.
    :param interval: time interval between frames in ms.
    """
    seq1 = seq1.permute(1, 2, 3, 0)
    seq2 = seq2.permute(1, 2, 3, 0)

    # calculate figure size based on image dimensions
    h, w = seq1.shape[1], seq1.shape[2]
    figsize = (w / 25, h / 25)

    # set up the figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(2 * figsize[0], figsize[1]))
    axes[0].axis("off")
    axes[1].axis("off")
    axes[0].set_title(title1)
    axes[1].set_title(title2)

    # initialize the image objects on the plots
    cmap = None
    if seq1.shape[-1] == 1:
        cmap = "gray"
    im1 = axes[0].imshow(seq1[0], cmap=cmap)
    im2 = axes[1].imshow(seq2[0], cmap=cmap)

    # create the animation
    def update(frame):
        im1.set_data(seq1[frame])
        im2.set_data(seq2[frame])
        return [im1, im2]

    _ = animation.FuncAnimation(
        fig, update, frames=min(len(seq1), len(seq2)), interval=interval, blit=True
    )

    plt.show()

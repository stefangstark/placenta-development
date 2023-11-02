from irtoolkit.viz import SlideSection
import seaborn as sns


def slide_1_region_J():
    fg_internal = SlideSection(
        slice(1000, 1200),
        slice(400, 500),
        color=sns.color_palette()[0],
        name='foreground - internal'
    )

    fg_external = SlideSection(
        slice(1100, 1250),
        slice(700, 800),
        color=sns.color_palette()[1],
        name='foreground - external'
    )

    bg = SlideSection(
        slice(400, 600),
        slice(700, 800),
        color=sns.color_palette()[2],
        name='background'
    )

    return [fg_internal, fg_external, bg]

def get_regions(stem):
    if stem == 'sample-slide-1-region-J':
        return slide_1_region_J()

    return

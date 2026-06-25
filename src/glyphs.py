IR3DE_BANNER = (
    " ▝▀▌▛▀  ▌▛▀▀▀▀▖ ▀▀▀▀▀▖ ▌▛▀▀▀▀▖ ▛▀▀▀▀▘\n"
    "   ▌▌   ▌▌    ▌      ▌ ▌▌    ▌ ▌▌    \n"
    "   ▌▌   ▌▛▀▀▀▚   ▀▀▀▚  ▌▌    ▌ ▛▀▀▀  \n"
    "   ▌▌   ▌▌    ▌      ▌ ▌▌   ▗▌ ▌▌    \n"
    " ▝▀▘▀▀  ▘▘    ▘ ▀▀▀▀▀  ▀▀▀▀▀▘  ▀▀▀▀▀▘"
)

DIAMOND_FRAMES = [
    # 0° — face-on
    "   ▄▄▄▄▄   \n"
    "  █     █  \n"
    " █       █ \n"
    "█         █\n"
    " █       █ \n"
    "  █     █  \n"
    "   ▀▀▀▀▀   ",

    # ~30° — slight compression
    "    ▄▄▄    \n"
    "   █   ▌   \n"
    "  █     ▌  \n"
    " █       ▌ \n"
    "  █     ▌  \n"
    "   █   ▌   \n"
    "    ▀▀▀    ",

    # ~60° — half compressed
    "     ▄     \n"
    "    █ ▌    \n"
    "   █   ▌   \n"
    "  █     ▌  \n"
    "   █   ▌   \n"
    "    █ ▌    \n"
    "     ▀     ",

    # ~80° — very thin
    "     ▄     \n"
    "     █     \n"
    "    █ ▌    \n"
    "   █   ▌   \n"
    "    █ ▌    \n"
    "     █     \n"
    "     ▀     ",

    # 90° — edge
    "     ▄     \n"
    "     █     \n"
    "     █     \n"
    "     █     \n"
    "     █     \n"
    "     █     \n"
    "     ▀     ",

    # ~100° — very thin
    "     ▄     \n"
    "     █     \n"
    "    ▐ █    \n"
    "   ▐   █   \n"
    "    ▐ █    \n"
    "     █     \n"
    "     ▀     ",

    # ~120° — half compressed
    "     ▄     \n"
    "    ▐ █    \n"
    "   ▐   █   \n"
    "  ▐     █  \n"
    "   ▐   █   \n"
    "    ▐ █    \n"
    "     ▀     ",

    # ~150° — slight compression
    "    ▄▄▄    \n"
    "   ▐   █   \n"
    "  ▐     █  \n"
    " ▐       █ \n"
    "  ▐     █  \n"
    "   ▐   █   \n"
    "    ▀▀▀    ",
]

# Visit each angle, then reverse back. One pass = 180° of rotation;
# two passes = a full revolution.
DIAMOND_ROTATION = [0, 1, 2, 3, 4, 5, 6, 7]
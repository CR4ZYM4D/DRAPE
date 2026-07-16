from typing import Dict, List

# types of attributes and their value lsists
GARMENT_TYPES = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit',
                  'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings',
                  'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper',
                  'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "indigo", "pink", "black", "white", "grey", "brown", "beige", "teal", "burgundy"]

CONTEXTS = ["office", "park", "street", "home", "gym", "beach", "restaurant"]

STYLES = ["minimalist", "tailored", "sporty", "relaxed", "dramatic", "bold", "elegant"]

# final tensor list
FLAT_ATTRIBUTE_LIST: List[str] = GARMENT_TYPES + COLORS + CONTEXTS + STYLES
TOTAL_DIMS = len(FLAT_ATTRIBUTE_LIST)

# slice mapping for breaking down the 1D prediction vector later
SCHEMA_SLICES: Dict[str, tuple] = {
    "garment_type": (0, len(GARMENT_TYPES)),
    "color": (len(GARMENT_TYPES), len(GARMENT_TYPES) + len(COLORS)),
    "context": (len(GARMENT_TYPES) + len(COLORS), len(GARMENT_TYPES) + len(COLORS) + len(CONTEXTS)),
    "style": (len(GARMENT_TYPES) + len(COLORS) + len(CONTEXTS), TOTAL_DIMS)
}

# lookup dictionary for mapping names to 1D tensor indices
ATTR_TO_IDX: Dict[str, int] = {attr: idx for idx, attr in enumerate(FLAT_ATTRIBUTE_LIST)}
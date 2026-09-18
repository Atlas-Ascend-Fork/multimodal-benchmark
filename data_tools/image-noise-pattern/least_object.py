from xml.dom.minidom import parse


XML_PATH = '../VOCdevkit/VOC2007/Annotations/'


def get_least_object_height_width(img_name):
    domTree = parse(XML_PATH + img_name.replace('jpg', 'xml'))
    # 文档根元素
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName('object')
    least_height = 0
    least_width = 0
    for object in objects:
        bndbox = object.getElementsByTagName('bndbox')
        for box in bndbox:
            xmin = int(box.childNodes[1].childNodes[0].data)
            ymin = int(box.childNodes[3].childNodes[0].data)
            xmax = int(box.childNodes[5].childNodes[0].data)
            ymax = int(box.childNodes[7].childNodes[0].data)
            if (ymax - ymin) < least_height or least_height == 0:
                least_height = ymax - ymin
            if (xmax - xmin) < least_width or least_width == 0:
                least_width = xmax - xmin
    return least_height, least_width

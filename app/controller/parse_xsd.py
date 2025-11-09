import xml.etree.ElementTree as ET

def parse_xsd(xsd_file):
    tree = ET.parse(xsd_file)
    root = tree.getroot()
    namespaces = {'xs': 'http://www.w3.org/2001/XMLSchema'}
    
    def get_elements_and_types(element):
        elements = []
        for child in element.findall('.//xs:element', namespaces):
            name = child.get('name')
            type_ = child.get('type')
            annotation = None
            annotation_tag = child.find('.//xs:annotation/xs:documentation', namespaces)
            if annotation_tag is not None:
                source = annotation_tag.get('source')
                if source and source.lower() == 'yellow field':
                    annotation = source
            if name and type_:
                elements.append((name, type_, annotation))
        return elements

    def collect_elements(element, path, elements_dict):
        elements_and_types = get_elements_and_types(element)
        for name, type_, annotation in elements_and_types:
            new_path = f"{path}/{name}"
            elements_dict[new_path] = (type_, annotation)
            complex_type = root.find(f".//xs:complexType[@name='{type_}']", namespaces)
            if complex_type is not None:
                collect_elements(complex_type, new_path, elements_dict)

    elements_dict = {}
    document_element = root.find(".//xs:element[@name='Document']", namespaces)
    if document_element is not None:
        document_type = document_element.get('type')
        document_path = '/Document'
        elements_dict[document_path] = (document_type, None)
        document_complex_type = root.find(f".//xs:complexType[@name='{document_type}']", namespaces)
        if document_complex_type is not None:
            collect_elements(document_complex_type, document_path, elements_dict)
    
    return elements_dict

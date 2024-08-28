import codecs


text = ""

# Encode the text using ROT13
rot13_encoded = codecs.encode(text, 'rot_13')

# Print the encoded text
print(rot13_encoded)

# Decode the text using ROT13
rot13_decoded = codecs.decode(rot13_encoded, 'rot_13')


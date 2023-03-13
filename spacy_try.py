import spacy

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    # text = "And deeper emissions cuts like those proposed by the European Union – 20% below 1990 levels within 12 years – would reduce global temperatures by only one-sixtieth of one degree Celsius (one-thirtieth of one degree Fahrenheit) by 2100, at a cost of $10 trillion."
    t = "And deeper emissions cuts like those proposed by the European Union – 20% below 1990 levels within 12 years – would reduce global temperatures by only one-sixtieth of one degree Celsius (one-thirtieth of one degree Fahrenheit) by 2100, at a cost of $10 trillion. For every dollar spent, we would do just four cents worth of good."
    t = "MADRID – On November 6, either Barack Obama or Mitt Romney will emerge victorious after an exhausting electoral race, setting the wheels in motion for the coming four years."
    piano_doc = nlp(t)

    for token in piano_doc:
        if token.dep_ == "ROOT":
            print(token.text)
            print(list(token.children))
    lior=5
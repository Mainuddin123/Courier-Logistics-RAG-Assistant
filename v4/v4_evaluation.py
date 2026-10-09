questions = [
    # Damaged / Lost Parcel
    "What should I do if my parcel is damaged?",
    "How should I document damage to a parcel?",
    "When should I report a damaged parcel?",
    "What evidence may be required for a damage claim?",
    "What should I do if my parcel is lost?",
    "What information should I provide for a lost parcel investigation?",

    # Address Change
    "Can I change my delivery address?",
    "Can I correct my address before dispatch?",
    "Can I change my address after dispatch?",
    "What happens if my delivery address is incorrect?",

    # Cancellation
    "How can I cancel my shipment?",
    "Can I cancel my order before dispatch?",
    "What should I do if I want to cancel a delivery?",
    "Is shipment cancellation always possible?",

    # Delivery
    "What should I do if there is a delivery problem?",
    "What happens if a delivery attempt fails?",
    "What information should I provide when contacting support about delivery?",

    # Prohibited Items
    "What items are prohibited?",
    "Can I send a prohibited item through the courier?",
    "What should I check before shipping an item?",
    
    # Return / Refund
    "How can I request a refund?",
    "What is the process for returning a parcel?",
    "What condition should I keep the parcel in for a return?",
    "When is a refund issued?",

    # Out-of-domain question
    "What is the company's employee leave policy?"
]


if __name__ == "__main__":

    print(f"Total evaluation questions: {len(questions)}")

    for i, question in enumerate(questions, start=1):
        print(f"{i}. {question}")
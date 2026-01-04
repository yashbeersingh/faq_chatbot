class ConversationMemory:
    def __init__(self):
        self.last_questions = []
        self.active_subcategory = None

    def update(self, question, subcategory=None):
        self.last_questions.append(question)
        self.last_questions = self.last_questions[-2:]
        if subcategory:
            self.active_subcategory = subcategory

import os
import pathlib
import random

os.environ['KERAS_BACKEND'] = 'torch'
import keras
import numpy
import abc


class Board(abc.ABC):
    @abc.abstractmethod
    def push(self, move) -> None:
        pass

    @abc.abstractmethod
    def undo(self) -> None:
        pass

    @abc.abstractmethod
    def finished(self) -> bool:
        pass

    @abc.abstractmethod
    def __repr__(self) -> bool:
        pass

    @abc.abstractmethod
    def has_won(self, side) -> bool:
        pass

    @staticmethod
    @abc.abstractmethod
    def flip_side(side):
        pass


class TicTacToe(Board):
    CROSS = 1
    CIRCLE = 0

    def __init__(self):
        self.board = numpy.zeros((3, 3, 2))
        self.moves = []

    def push(self, move):
        piece, position = move
        self.board[*position, piece] = 1
        self.moves.append((position, piece))

    def undo(self):
        if len(self.moves):
            position, piece = self.moves.pop(-1)
            self.board[*position, piece] = 0

    def has_won(self, piece):
        board = self.board[:, :, piece]
        if numpy.all(numpy.diag(board)) or numpy.all(numpy.diag(numpy.fliplr(board))):
            return True
        for i in range(3):
            if numpy.all(board[i]) or numpy.all(board[:, i]):
                return True
        return False

    def finished(self):
        return self.has_won(TicTacToe.CROSS) or self.has_won(TicTacToe.CIRCLE) or numpy.all(
            numpy.any(self.board, axis=2))

    def get_moves(self):
        return numpy.column_stack(numpy.where(numpy.any(self.board, axis=2) == 0))

    def __repr__(self):
        string = ''
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j][TicTacToe.CROSS]:
                    string += 'X'
                elif self.board[i][j][TicTacToe.CIRCLE]:
                    string += 'O'
                else:
                    string += '_'
                string += ' '
            string += '\n'
        return string

    @staticmethod
    def flip_side(side):
        return 1 - side


class Simulation:
    def __init__(self, board: type):
        self._board = board()
        self.states = []

    def finished(self):
        return self._board.finished()

    def get_permutations(self, side):
        moves = self._board.get_moves()
        for i in range(len(moves)):
            self._board.push((side, moves[i]))
            state = game.board if side == self.game.CROSS else game.board[:, :, ::-1]
            score = self.model.predict(numpy.expand_dims(state, axis=0), verbose=False)
            values[i] = score.item()
            game.undo()

    def push(self, move):
        self._board.push(move)
        self.states.append(self._board.copy())


class DMCTS:
    def __init__(self, game: type, model=None):
        self.game = game
        self.model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=2, input_shape=(3, 3, 2), activation='relu'),
            keras.layers.Conv2D(filters=64, kernel_size=2, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        if model is not None:
            self.model.load_weights(model)

    def play(self):
        game = self.game()
        while not game.finished():
            moves = game.get_moves()
            values = numpy.zeros(len(moves))
            for i in range(len(moves)):
                game.push(Board.CROSS, moves[i])
                score = self.model.predict(numpy.expand_dims(game.board, axis=0), verbose=False)
                values[i] = score.item()
                game.undo()
            best = numpy.argmax(values)
            game.push(Board.CROSS, moves[best])
            print(game)
            if game.finished():
                break
            user = (int(input('Row: ')), int(input('Col: ')))
            game.push(Board.CIRCLE, user)
            print(game)

    def train(self, simulations, epochs):
        for epoch in range(epochs):
            q = {}

            for simulation in range(simulations):
                states = []
                side = self.game.CROSS
                game = self.game()
                while not game.finished():
                    moves = game.get_moves()
                    values = numpy.zeros(len(moves))
                    for i in range(len(moves)):
                        game.push((side, moves[i]))
                        state = game.board if side == self.game.CROSS else game.board[:, :, ::-1]
                        score = self.model.predict(numpy.expand_dims(state, axis=0), verbose=False)
                        values[i] = score.item()
                        game.undo()
                    if side == self.game.CROSS:
                        best = numpy.argmax(values)
                    else:
                        best = numpy.argmin(values)
                    if random.random() <= 0.2:
                        best = int(random.random() * len(moves))
                    game.push((side, moves[best]))
                    states.append(game.board.copy())
                    side = game.flip_side(side)

                reward = 0
                if game.has_won(self.game.CROSS):
                    reward = 1
                elif game.has_won(self.game.CIRCLE):
                    reward = -1

                for state in states:
                    hashed = state.tobytes()
                    if hashed not in q:
                        q[hashed] = []
                    q[hashed].append(reward)

            keys = list(q.keys())
            values = [numpy.average(numpy.array(q[key])) for key in keys]

            train_x = numpy.array([numpy.frombuffer(key).reshape((3, 3, 2)) for key in keys])
            train_y = numpy.array(values)

            self.model.fit(train_x, train_y)

        directory = pathlib.Path('models')
        directory.mkdir(exist_ok=True)
        number = 0
        while (directory / f'model{number}.keras').exists():
            number += 1
        path = directory / f'model{number}.keras'
        self.model.save(path)


if __name__ == '__main__':
    path = None
    #path = os.path.join('models', 'model4.keras')
    model = DMCTS(TicTacToe, path)
    model.model.summary()
    model.train(500, 25)
    #model.play()

class TulipStrategy(object):
    """Mealy transducer.

    Internal states are integers, the current state
    is stored in the attribute "state".
    To take a transition, call method "move".

    The names of input variables are stored in the
    attribute "input_vars".

    Automatically generated by tulip.dumpsmach on 2023-10-21 01:07:12 UTC
    To learn more about TuLiP, visit http://tulip-control.org
    """
    def __init__(self):
        self.state = 5
        self.input_vars = ['xped']

    def move(self, xped):
        """Given inputs, take move and return outputs.

        @rtype: dict
        @return: dictionary with keys of the output variable names:
            ['xcar', 'vcar']
        """
        output = dict()
        if self.state == 0:
            if (xped == 0):
                self.state = 3

                output["xcar"] = 2
                output["vcar"] = 1
            else:
                self._error(xped)
        elif self.state == 1:
            if (xped == 1):
                self.state = 2

                output["xcar"] = 2
                output["vcar"] = 0
            else:
                self._error(xped)
        elif self.state == 2:
            if (xped == 1):
                self.state = 2

                output["xcar"] = 2
                output["vcar"] = 0
            else:
                self._error(xped)
        elif self.state == 3:
            if (xped == 0):
                self.state = 4

                output["xcar"] = 3
                output["vcar"] = 0
            else:
                self._error(xped)
        elif self.state == 4:
            if (xped == 0):
                self.state = 4

                output["xcar"] = 3
                output["vcar"] = 0
            else:
                self._error(xped)
        elif self.state == 5:
            if (xped == 0):
                self.state = 0

                output["xcar"] = 1
                output["vcar"] = 1
            elif (xped == 1):
                self.state = 1

                output["xcar"] = 1
                output["vcar"] = 1
            else:
                self._error(xped)
        else:
            raise Exception("Unrecognized internal state: " + str(self.state))
        return output

    def _error(self, xped):
        raise ValueError("Unrecognized input: " + (
            "xped = {xped}; ").format(
                xped=xped))

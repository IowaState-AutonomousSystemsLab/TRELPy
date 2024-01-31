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
        self.input_vars = ['xobj']

    def move(self, xobj):
        """Given inputs, take move and return outputs.

        @rtype: dict
        @return: dictionary with keys of the output variable names:
            ['xcar', 'vcar']
        """
        output = dict()
        if self.state == 0:
            if (xobj == 1):
                self.state = 1

                output["xcar"] = 2
                output["vcar"] = 1
            else:
                self._error(xobj)
        elif self.state == 1:
            if (xobj == 1):
                self.state = 2

                output["xcar"] = 3
                output["vcar"] = 1
            else:
                self._error(xobj)
        elif self.state == 2:
            if (xobj == 1):
                self.state = 3

                output["xcar"] = 4
                output["vcar"] = 1
            else:
                self._error(xobj)
        elif self.state == 3:
            if (xobj == 1):
                self.state = 4

                output["xcar"] = 5
                output["vcar"] = 1
            else:
                self._error(xobj)
        elif self.state == 4:
            if (xobj == 1):
                self.state = 4

                output["xcar"] = 5
                output["vcar"] = 1
            else:
                self._error(xobj)
        elif self.state == 5:
            if (xobj == 1):
                self.state = 0

                output["xcar"] = 1
                output["vcar"] = 1
            else:
                self._error(xobj)
        else:
            raise Exception("Unrecognized internal state: " + str(self.state))
        return output

    def _error(self, xobj):
        raise ValueError("Unrecognized input: " + (
            "xobj = {xobj}; ").format(
                xobj=xobj))
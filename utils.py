
#########################################################################
# Used for better printing outputs
class _terminal_output:
    """ Provides basic functionality to create a string that contains
    formatting to style the terminal output, such as when using the print
    function.
    
    Class can be used by itself, but is a sub-class of print2, which 
    provides more functionality to produce print outputs.
    
    Explanation:
     * The terminal can apply styles using the format
        escape style end
       i.e. "\033[1m" makes the text bold, where
         escape = "\033["
         style  = "1"
         end    = "m"
     * To end (or reset) the formatting, a reset style should be applied
       i.e. "\033[0m" resets the style
     * Multiple styles can be joined, using a semi-colan seperator
       i.e. "\033[1;31m" makes the text bold and red.
     * You should therefore apply a style, add a string (message) and then 
       reset the style
       i.e. "\033[1mI AM A BOLD MESSAGE\033[0m"
    
    Usage:
     >> new_style(style) converts a style into the desired format
        i.e. new_style(1) produces "\033[1m"
     >> end_style() ends (or resets) the applied style
     >> create_style(style   = string or tuple of style(s),
                     message = the string to insert after the style
                     ending  = Boolean to confirm if style should be reset
        i.e create_style(style = (1, 31), "hello world", ending = True) would
        return "\033[1;31mhello world\033[0m"
    """
    _escape = "\033["
    _end    = "m"
    _sep    = ";"

    class style:
        reset      = "0"
        bold       = "1"
        light      = "2"
        italic     = "3"
        underlined = "4"

    class fg_color:
        black        = "30"
        red          = "31"
        green        = "32"
        yellow       = "33"
        blue         = "34"
        purple       = "35"
        cyan         = "36"
        white        = "37"
        dark_grey    = "90"
        bright_red    = "91"
        bright_green  = "92"
        bright_yellow = "93"
        bright_blue   = "94"
        bright_purple = "95"
        bright_cyan   = "96"
        bright_white  = "97"
    
    class bg_color:
        black        = "40"
        red          = "41"
        green        = "42"
        yellow       = "43"
        blue         = "44"
        purple       = "45"
        cyan         = "46"
        white        = "47"
        dark_grey    = "100"
        bright_red    = "101"
        bright_green  = "102"
        bright_yellow = "103"
        bright_blue   = "104"
        bright_purple = "105"
        bright_cyan   = "106"
        bright_white  = "107"

    @classmethod
    def new_style(self, style):
        if type(style) is tuple: style = self._join(style)
        return self._escape + style + self._end

    @classmethod
    def end_style(self):
        return self._escape + self.style.reset + self._end

    @classmethod
    def _join(self, styles):
        new_style = ""
        for style in styles:
            if new_style != "": new_style = new_style + self._sep
            new_style = new_style + style
        return new_style

    @classmethod
    def create_style(self, style, message = "", ending = True):
        if type(style) is tuple: style = self._join(style)
        output = self.new_style(style) + message
        if ending: output = output + self.end_style()
        return output


class print2:
    """ Provides an extended print functionality for IDE's/Code Editors that 
    support formatting. Supports any text encoding, however print2.formatting.* 
    should be used for ease. Styles can be a single string (one style) or a 
    tuple for many combined styles.
    
    Usage:
    >>> print2.heading(string, **kwargs)
    >>> print2.warning(string, **kwargs)
    >>> print2.print(string, style = print2.fg_color.purple)
    >>> print2.print(string, style = print2.bg_color.white, 
                     _print = False, _return = True)
    """
    

    class formatting(_terminal_output):pass
    
    @classmethod
    def print(self, string, style, _print = True, _return = False, end = "\n"):
        """
        string  = the string / message to print
        style   = the style to apply (use print2.formatting.* for ease)
        _print  = print the result to console
        _return = return the entire output as a string
        
        Usage:
         >> print2.print("I am a message",
                         style = print2.formatting.style.underlined)
        """
        output = _terminal_output.create_style(style, string)

        if _print:
            print(output, end = end)
        if _return:
            _return = output
        else:
            _return = None
        return _return
    @classmethod
    def bold(self, string, **kwargs):       return self.print(string, style = self.formatting.style.bold, **kwargs)
    @classmethod
    def underlined(self, string, **kwargs): return self.print(string, style = self.formatting.style.underlined, **kwargs)
    @classmethod
    def heading(self, string, **kwargs):    return self.print(string, style = (self.formatting.style.bold, self.formatting.style.underlined), **kwargs)
    @classmethod
    def warning(self, string, **kwargs):    return self.print("   WARNING: " + string + "   ", style = (self.formatting.bg_color.red, self.formatting.style.bold), **kwargs)
    @classmethod
    def warning2(self, string, **kwargs):   return self.print(string, style = (self.formatting.fg_color.red, self.formatting.style.bold))
    @classmethod
    def note(self, string, **kwargs):       return self.print(self.print("NOTE: ", style = self.formatting.style.bold, _print = False, _return = True) + string, style = "0", **kwargs)
    @classmethod
    def highlight(self, string, fg_color = None, bg_color = None, margin = " ", **kwargs):
        if fg_color is None: fg_color = self.formatting.fg_color.black
        if bg_color is None: bg_color = self.formatting.bg_color.white
        return self.print(margin + string + margin, style = (self.formatting.style.bold, fg_color, bg_color), **kwargs)


#########################################################################



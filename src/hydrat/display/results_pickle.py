from cPickle import dump, load

class results_pickle():
  def __init__(self, target, real_renderer, start_depth = 0):
    self.target = target
    self.commands = []
    self.depth = start_depth
    self.renderer = real_renderer

  def __repr__(self): return "<Pickled Results>"

  def __getattr__(self, key):
    #print "GETTING ", key
    def recorded_execution(*args, **kwargs):
      self.commands.append((True, (key, args, kwargs)))
      self.target.seek(0)
      dump(self.commands, self.target)
      self.target.truncate()
      if key == "__nonzero__": 
        return True
      try:
        return getattr(self.renderer, key).__call__(*args, **kwargs)
      except AttributeError, e:
        print e
        print key
        print args
        print kwargs
        raise Exception

    if key in self.__dict__:
      return self.__dict__[key]
    else:
      return recorded_execution 

  def __setattr__(self,name,value):
    #print "SET %s to %s" % (str(name),str(value))
    if name == "depth" :
      self.commands.append((False, (name, value)))
    self.__dict__[name] = value
    self.renderer.__dict__[name] = value

class pickled_result_renderer():
  def __init__(self, renderer, pickled_steps):
    self.renderer = renderer
    self.commands = load(pickled_steps) 
    
  def __call__(self):
    for command in self.commands:
      self.render(command)

  def render(self, step):
    isCall, params = step
    if isCall:
      name, args, kwargs = params
      getattr(self.renderer, name).__call__(*args, **kwargs)
    else:
      key, value = params
      setattr(self.renderer, key, value)

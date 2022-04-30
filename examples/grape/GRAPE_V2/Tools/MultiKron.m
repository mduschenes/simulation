function A = MultiKron(varargin)
    A = varargin{1};
    for k=2:length(varargin)
        A = kron(A,varargin{k});
    end

end
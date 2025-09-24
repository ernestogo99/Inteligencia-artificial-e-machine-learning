def newtow_raphson(function,function_prime,x0,tol=1e-6,max_iter=100):
    x=x0
    for i in range(max_iter):
        fx= function(x)
        if abs(fx)<tol:
            print(f"Convergência atingida em {i} iterações.")
            return x
        
        x= x-fx/function_prime(x)
    raise ValueError("Não convergiu dentro do número máximo de iterações.")
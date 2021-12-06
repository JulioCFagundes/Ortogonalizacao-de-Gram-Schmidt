using LinearAlgebra

"""
    Q, R = gram_schmidt(A)
O processo de Gram-Schmidt foi estudado em Álgebra Linear, e consiste em ortogonalizar ou ortonormalizar vetores.
Aqui iremos ortonormalizar cada coluna da matriz A, e armazenar os coeficientes utilizados para isso.
Notação:
- aⱼ é a j-ésima coluna da matriz j.
- m é o número de linhas
- n é o número de colunas
Primeiro, fazemos a ortogonalização:
    v₁ = a₁
    
    v₂ = a₂ - α₁₂ v₁
    v₃ = a₃ - α₁₃ v₁ - α₂₃ v₂
    ...
onde
    αᵢⱼ = dot(vᵢ, aⱼ) / dot(vᵢ, vᵢ),    1 ≤ i < j ≤ n
Depois, normalizamos cada vetor.
    qᵢ = vᵢ / ‖vᵢ‖
onde qᵢ é a i-ésima coluna da matriz Q m×n.
A matriz R n×n é quadrada triangular superior com cada elemento dada por
    rⱼⱼ = ‖vⱼ‖
    rᵢⱼ = αᵢⱼ × ‖vᵢ‖,   1 ≤ i < j ≤ n
Se m < n o método deve gerar um erro.
"""
function gram_schmidt(A)
    m,n = size(A)
    if m < n
        error("matriz deve ser quadrada ou ter mais linhas que colunas meu chapa")
    end
    Q = zeros(m,n)
    R = zeros(n,n)
    v = zeros(m,n)
    v[:,1] = A[:,1]
    #Ortogonalização
    for j in 2:n
        aⱼ = A[:,j]
        v[:,j] = aⱼ
        for i in 1:j-1
            aᵢⱼ = dot(v[:,i], aⱼ) / dot(v[:,i], v[:,i])
            v[:,j] = v[:,j] - aᵢⱼ * v[:,i]
        end
    end
    #Normalização
    for j in 1:n
        Q[:,j] = v[:,j] / sqrt(dot(v[:,j], v[:,j]))
        R[j,j] = sqrt(dot(v[:,j], v[:,j]))
        aⱼ = A[:,j]
        for i in 1:j-1
            aᵢⱼ = dot(v[:,i], aⱼ) / dot(v[:,i], v[:,i])
            R[i,j] = aᵢⱼ * sqrt(dot(v[:,i], v[:,i]))
        end
    end
    return Q, R
end
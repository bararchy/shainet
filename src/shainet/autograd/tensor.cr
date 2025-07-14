module SHAInet
  module Autograd
    class Tensor
      # Use Float32 for data and gradient storage to reduce memory
      # footprint and encourage fp32 computations.
      property data : Float32
      property grad : Float32
      property parents : Array(Tensor)
      property backward_fn : Proc(Float32, Nil)?

      def initialize(@data : Float32, @parents : Array(Tensor) = [] of Tensor, @backward_fn : Proc(Float32, Nil)? = nil)
        @grad = 0_f32
      end

      def +(other : Tensor)
        Tensor.new(@data + other.data, [self, other], ->(g : Float32) do
          self.grad += g
          other.grad += g
        end)
      end

      def +(other : Number)
        self + Tensor.new(other.to_f32)
      end

      def -(other : Tensor)
        Tensor.new(@data - other.data, [self, other], ->(g : Float32) do
          self.grad += g
          other.grad -= g
        end)
      end

      def -(other : Number)
        self - Tensor.new(other.to_f32)
      end

      def *(other : Tensor)
        Tensor.new(@data * other.data, [self, other], ->(g : Float32) do
          self.grad += g * other.data
          other.grad += g * self.data
        end)
      end

      def *(other : Number)
        self * Tensor.new(other.to_f32)
      end

      def /(other : Tensor)
        Tensor.new(@data / other.data, [self, other], ->(g : Float32) do
          self.grad += g / other.data
          other.grad -= g * @data / (other.data * other.data)
        end)
      end

      def /(other : Number)
        self / Tensor.new(other.to_f32)
      end

      def matmul(other : Tensor)
        Tensor.new(@data * other.data, [self, other], ->(g : Float32) do
          self.grad += g * other.data
          other.grad += g * self.data
        end)
      end

      def backward(initial_grad : Float32 = 1_f32)
        build_topology(self, Set(Tensor).new).reverse_each do |t|
          if t == self
            t.grad += initial_grad
          end
          t.backward_fn.try &.call(t.grad)
        end
      end

      private def build_topology(t : Tensor, visited : Set(Tensor), topo = [] of Tensor)
        unless visited.includes?(t)
          visited.add(t)
          t.parents.each { |p| build_topology(p, visited, topo) }
          topo << t
        end
        topo
      end
    end
  end
end

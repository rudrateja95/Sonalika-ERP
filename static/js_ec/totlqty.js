async function loadCartCount() {

    const res = await fetch("/api/cart-count");

    const data = await res.json();

    document.getElementById("cartQty").textContent = data.total_qty;
}

loadCartCount();